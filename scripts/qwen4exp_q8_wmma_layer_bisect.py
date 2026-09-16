#!/usr/bin/env python3
"""Bracket the Q8 dense WMMA prefill route's admissible layer scope.

The route recovers prefill time in proportion to the Q8_0 bytes its layer scope
owns, and it fails the calibrated numerical envelope somewhere between layers 0
and 32.  This finds that boundary with cheap screens and leaves certification to
the full gate, following the two-tier rule in ``benchmarks/HARNESSES.md``: never
bracket a bound with full-cost points.

A **screen** is the same gate at a reduced protocol - one prompt per category at
the shortest length, fewer decode steps - so it costs roughly a twelfth of a
full arm.  A screen never admits a scope.  It only says which side of the
boundary a scope is on, so the full gate can be spent once, on the winner.

Probe order is value-weighted rather than a plain midpoint, because the route's
value is concentrated at the ends of the layer axis: layers 28-31 and 0-7 are
each worth about 1 s on the measured prefill while all twenty layers 8-27
together are worth about 1.3 s.  A midpoint bisection would spend arms
separating scopes that differ by a quarter of a second.

Usage::

    # screen specific scopes
    python3 scripts/qwen4exp_q8_wmma_layer_bisect.py --model-root PATH \
        --probe 28 --probe 16 --scratch DIR

    # let the ledger choose the next probe by expected value
    python3 scripts/qwen4exp_q8_wmma_layer_bisect.py --model-root PATH \
        --auto --scratch DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "execution_profile_q8_wmma_prefill_layers_gate.py"
DEFAULT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"
COVERAGE = (
    ROOT
    / "benchmarks"
    / "results"
    / "2026-09-16-q8-wmma-dense-prefill-layers-gate"
    / "route-coverage.json"
)
# One prompt per category at the shortest length. general_en must be present:
# it carries the tail and every top-1 miss in the 32-47 arm, so a screen without
# it would be blind to the failure mode that matters.
SCREEN_CASES = (
    "code-p512",
    "general_en-p512",
    "general_ja-p512",
    "mixed_ja_en-p512",
)
SCREEN_DECODE_STEPS = 32
# The evaluator requires at least three candidate repeats, and the repeat
# determinism check is worth keeping even in a screen.
SCREEN_REPEAT_RUNS = 3
LAYER_COUNT = 48
# The measured saving at full scope on code-p4096, gfx1151. Used only to weight
# probe order and to report what a bracket is worth; it is not a claim.
FULL_SCOPE_SAVED_S = 7.0409


def _scope_label(start: int) -> str:
    return f"{start}-{LAYER_COUNT - 1}"


def _route_bytes_by_layer() -> dict[int, int]:
    if not COVERAGE.is_file():
        return {}
    payload = json.loads(COVERAGE.read_text(encoding="utf-8"))
    return {
        int(layer): sum(int(value) for value in roles.values())
        for layer, roles in payload.get("in_block", {}).items()
    }


def _predicted_saving(start: int, by_layer: dict[int, int]) -> float | None:
    if not by_layer:
        return None
    total = sum(by_layer.values())
    if not total:
        return None
    owned = sum(by_layer.get(layer, 0) for layer in range(start, LAYER_COUNT))
    return FULL_SCOPE_SAVED_S * owned / total


def run_screen(
    start: int,
    *,
    model_root: Path,
    fixture: Path,
    scratch: Path,
    python: str,
) -> dict[str, Any]:
    output = scratch / f"screen-layers{start}-{LAYER_COUNT - 1}.json"
    command = [
        python,
        str(GATE),
        "--model-root", str(model_root),
        "--fixture", str(fixture),
        "--layers", _scope_label(start),
        "--decode-steps", str(SCREEN_DECODE_STEPS),
        "--repeat-runs", str(SCREEN_REPEAT_RUNS),
        "--prefill-chunk-size", "1024",
        "--output", str(output),
    ]
    for case_id in SCREEN_CASES:
        command += ["--case-id", case_id]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if not output.is_file():
        return {
            "scope": _scope_label(start),
            "start": start,
            "ok": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-1200:],
        }
    payload = json.loads(output.read_text(encoding="utf-8"))
    quality = payload.get("quality", {})
    summary = quality.get("summary", {})
    return {
        "scope": _scope_label(start),
        "start": start,
        "ok": True,
        "screen": True,
        "artifact": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "rows": summary.get("rows"),
        "kl_mean": summary.get("kl_mean"),
        "kl_p95": summary.get("kl_p95"),
        "kl_max": summary.get("kl_max"),
        "top1_agreement": summary.get("top1_agreement"),
        "flip_eligible_share": summary.get("flip_eligible_share"),
        "max_abs_logit_delta": summary.get("max_abs_logit_delta"),
        "hard_gates_passed": quality.get("hard_gates_passed"),
        "repeat_deterministic": payload.get("repeat_determinism", {}).get("passed"),
        "scope_failures": quality.get("scope_failures", []),
        "thresholds": quality.get("thresholds", {}),
    }


def seed_from_full_gate(results_dir: Path) -> list[dict[str, Any]]:
    """Adopt the full-gate arms already on disk as known bracket endpoints.

    A screen only has to resolve scopes the full gate has not already settled,
    and a full arm outranks a screen wherever both exist.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("artifact*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("kind") != "hipengine_execution_profile_q8_wmma_prefill_layers_gate":
            continue
        layers = payload.get("route", {}).get("candidate_layers", "")
        values = sorted(int(part) for part in layers.split(",") if part.strip())
        if not values or values != list(range(values[0], LAYER_COUNT)):
            continue
        quality = payload.get("quality", {})
        summary = quality.get("summary", {})
        rows.append({
            "scope": _scope_label(values[0]),
            "start": values[0],
            "ok": True,
            "screen": False,
            "artifact": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "rows": summary.get("rows"),
            "kl_mean": summary.get("kl_mean"),
            "kl_p95": summary.get("kl_p95"),
            "kl_max": summary.get("kl_max"),
            "top1_agreement": summary.get("top1_agreement"),
            "flip_eligible_share": summary.get("flip_eligible_share"),
            "max_abs_logit_delta": summary.get("max_abs_logit_delta"),
            "hard_gates_passed": quality.get("hard_gates_passed"),
            "repeat_deterministic": payload.get("repeat_determinism", {}).get("passed"),
            "scope_failures": quality.get("scope_failures", []),
        })
    return rows


def _screened_side(row: dict[str, Any]) -> str | None:
    if not row.get("ok"):
        return None
    return "pass" if row.get("hard_gates_passed") else "fail"


def next_probe(ledger: Sequence[dict[str, Any]], by_layer: dict[int, int]) -> int | None:
    """Pick the probe that resolves the most unclaimed time per arm.

    Candidates are multiples of four, because the Q8_0 role layout is periodic
    with period four: a boundary off that grid changes the arm's attention-role
    composition as well as its depth, confounding the two.
    """

    passing = [row["start"] for row in ledger if _screened_side(row) == "pass"]
    failing = [row["start"] for row in ledger if _screened_side(row) == "fail"]
    # Known from the full gate: 32-47 passes, 0-47 fails.
    low = max(failing) if failing else -1      # deepest known failure
    high = min(passing) if passing else LAYER_COUNT  # shallowest known pass
    candidates = [
        start
        for start in range(0, LAYER_COUNT, 4)
        if low < start < high
    ]
    if not candidates:
        return None
    # Bisect on cumulative value rather than on layer index. Halving the layer
    # range would spend arms separating scopes a quarter of a second apart in
    # the cheap middle band, while halving the value range puts probes where the
    # unclaimed time actually is. Falls back to an index midpoint when the
    # coverage map is unavailable.
    low_value = _predicted_saving(max(low, 0), by_layer)
    high_value = _predicted_saving(min(high, LAYER_COUNT), by_layer)
    if low_value is None or high_value is None:
        return candidates[len(candidates) // 2]
    target = (low_value + high_value) / 2.0
    return min(
        candidates,
        key=lambda start: abs((_predicted_saving(start, by_layer) or 0.0) - target),
    )


def render(ledger: Sequence[dict[str, Any]], by_layer: dict[int, int]) -> str:
    lines = ["# Screen ledger (screens bracket; they never admit a scope)", ""]
    header = ["scope", "rows", "mean KL", "p95 KL", "top-1", "screen verdict", "pred saved s"]
    rows = []
    for row in sorted(ledger, key=lambda item: item["start"]):
        if not row.get("ok"):
            rows.append([row["scope"], "-", "-", "-", "-", "ERROR", "-"])
            continue
        predicted = _predicted_saving(row["start"], by_layer)
        rows.append([
            row["scope"],
            str(row.get("rows")),
            f"{row['kl_mean']:.3e}" if row.get("kl_mean") is not None else "-",
            f"{row['kl_p95']:.3e}" if row.get("kl_p95") is not None else "-",
            f"{row['top1_agreement']:.5f}" if row.get("top1_agreement") is not None else "-",
            ("pass" if row.get("hard_gates_passed") else "FAIL")
            + ("" if row.get("screen", True) else " (full gate)"),
            f"{predicted:.3f}" if predicted is not None else "-",
        ])
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    def line(cells): 
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    lines.append(line(header))
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    lines += [line(r) for r in rows]

    passing = [r["start"] for r in ledger if _screened_side(r) == "pass"]
    failing = [r["start"] for r in ledger if _screened_side(r) == "fail"]
    lines.append("")
    if passing:
        best = min(passing)
        predicted = _predicted_saving(best, by_layer)
        lines.append(
            f"Deepest screened pass: layers {_scope_label(best)}"
            + (f" (~{predicted:.3f} s predicted)" if predicted is not None else "")
            + " - certify with the full gate before claiming it."
        )
    if failing:
        lines.append(f"Shallowest screened fail: layers {_scope_label(max(failing))}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--probe", action="append", type=int, default=None,
        help="layer-scope start to screen (repeatable)",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="choose the next probe from the ledger by expected value",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="with --auto, how many probes to run",
    )
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args(argv)

    args.scratch.mkdir(parents=True, exist_ok=True)
    ledger_path = args.ledger or (args.scratch / "ledger.json")
    ledger: list[dict[str, Any]] = (
        json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else []
    )
    # Full-gate arms on disk outrank any screen for the same scope.
    known = seed_from_full_gate(COVERAGE.parent)
    known_starts = {row["start"] for row in known}
    ledger = known + [row for row in ledger if row["start"] not in known_starts]
    by_layer = _route_bytes_by_layer()

    probes = list(args.probe or [])
    if args.auto:
        probes += [-1] * int(args.rounds)

    for probe in probes:
        start = probe
        if start < 0:
            start = next_probe(ledger, by_layer)
            if start is None:
                print("no candidate probe remains; the bracket is closed")
                break
        predicted = _predicted_saving(start, by_layer)
        print(
            f"=== screening layers {_scope_label(start)}"
            + (f" (~{predicted:.3f} s predicted) ===" if predicted is not None else " ==="),
            flush=True,
        )
        row = run_screen(
            start,
            model_root=args.model_root.resolve(),
            fixture=args.fixture.resolve(),
            scratch=args.scratch,
            python=args.python,
        )
        ledger = [item for item in ledger if item["start"] != start] + [row]
        ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
        verdict = "ERROR" if not row.get("ok") else (
            "pass" if row.get("hard_gates_passed") else "FAIL"
        )
        print(
            f"  {verdict}"
            + (
                f"  mean KL {row['kl_mean']:.3e}  p95 {row['kl_p95']:.3e}"
                f"  top-1 {row['top1_agreement']:.5f}  rows {row['rows']}"
                if row.get("ok") else f"  rc={row.get('returncode')}"
            ),
            flush=True,
        )

    print()
    print(render(ledger, by_layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
