"""Roll the per-arm suite passes of the prefill-schedule probe into one artifact.

``scripts/tp2_prefill_schedule_failure_probe.py`` runs one arm at a time across
the teacher suite (``--prompts all``), because holding two sessions at once
doubles the resident footprint per rank. This script merges those passes and
derives the two numbers the sustained gate needs:

* **authorized horizon** — the largest depth ``D`` whose every prefix satisfies
  the declared production envelope on every prompt, per arm. ``D=128`` is not a
  defect discriminator if no route reaches it.
* **first breaching depth per prompt** — the decode index at which each prompt
  leaves the envelope, per arm. If materially different implementations breach
  the *same* prompt at the *same* depth, the depth is a property of the
  reference trajectory rather than of any candidate's arithmetic.

It also reports how flat the teacher's distribution is at the positions that
breach, because a near-tie row turns a residual logit difference into a large KL
while leaving the argmax unchanged.

Diagnostics only: no performance claim, and nothing is promoted.

Usage::

    python3 scripts/tp2_prefill_schedule_suite_rollup.py \\
        --teacher /path/teacher.json \\
        --pass tp1-serial=/path/tp2-suite-tp1.json \\
        --pass tp2-serial=/path/tp2-suite-tp2-serial.json \\
        --pass tp2-bulk=/path/tp2-suite-tp2-bulk.json \\
        --json benchmarks/results/<artifact>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tp2_teacher_coverage_broad import PRODUCTION_GATE  # noqa: E402

#: Below this depth the percentile bars are computed from too few rows to mean
#: anything (a p99 over two rows is the maximum), so horizons start here.
HORIZON_FLOOR = 8

#: A row this far from the threshold is worth reporting as "large"; the probe
#: uses the same 0.01 for its own count.
LARGE_KL = 0.01


def _envelope_failures(kl: np.ndarray) -> list[str]:
    """Which declared envelope statistics this block of rows violates."""

    failures = []
    if kl.size == 0:
        return ["no rows"]
    for key, statistic in (
        ("mean_kl", lambda values: values.mean()),
        ("p95_kl", lambda values: np.percentile(values, 95)),
        ("p99_kl", lambda values: np.percentile(values, 99)),
        ("max_kl", lambda values: values.max()),
    ):
        if float(statistic(kl)) > PRODUCTION_GATE[key]:
            failures.append(key)
    return failures


def _authorized_horizon(curves: dict[str, np.ndarray], floor: int = HORIZON_FLOOR) -> dict:
    """Largest depth whose every prefix passes on every prompt, plus the first miss."""

    depth = floor
    first_failure = None
    for candidate in range(floor + 1, max(len(curve) for curve in curves.values()) + 1):
        offenders = {
            prompt_id: _envelope_failures(curve[:candidate])
            for prompt_id, curve in curves.items()
        }
        offenders = {name: why for name, why in offenders.items() if why}
        if offenders:
            first_failure = {"depth": candidate, "prompts": offenders}
            break
        depth = candidate
    return {
        "floor": floor,
        "authorized_horizon": depth,
        "first_failure": first_failure,
    }


def _first_breaching_depth(curve: np.ndarray, floor: int = HORIZON_FLOOR) -> int | None:
    for candidate in range(floor + 1, len(curve) + 1):
        if _envelope_failures(curve[:candidate]):
            return candidate
    return None


def _distribution_shape(row: np.ndarray) -> tuple[float, float]:
    """Top-1 probability and top-2 logit gap of one row."""

    values = np.asarray(row, dtype=np.float64)
    ordered = np.sort(values)
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return float(probabilities.max()), float(ordered[-1] - ordered[-2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument(
        "--pass",
        dest="passes",
        action="append",
        required=True,
        metavar="ARM=JSON",
        help="one probe suite pass; repeat per arm",
    )
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--horizon-floor", type=int, default=HORIZON_FLOOR)
    args = parser.parse_args(argv)

    teacher = json.loads(args.teacher.read_text())
    teacher_sha = hashlib.sha256(args.teacher.read_bytes()).hexdigest()

    passes: dict[str, dict] = {}
    revisions: set[str] = set()
    for spec in args.passes:
        if "=" not in spec:
            raise ValueError(f"--pass expects ARM=JSON, got {spec!r}")
        arm, _, path = spec.partition("=")
        artifact = json.loads(Path(path).read_text())
        if artifact.get("kind") != "tp2_prefill_schedule_failure_probe":
            raise ValueError(f"{path} is not a prefill-schedule probe artifact")
        if artifact.get("teacher_sha256") != teacher_sha:
            raise ValueError(f"{path} scored against a different teacher capture")
        revisions.add(json.dumps(artifact.get("source_sha256"), sort_keys=True))
        passes[arm.strip()] = {"path": path, "artifact": artifact}
    if len(revisions) > 1:
        # Arms are run in separate processes, so a mixed set can be stitched
        # together silently. Each pass must come from the same arithmetic.
        raise ValueError(
            "passes come from different source revisions; rerun them against one "
            "frozen tree (scripts/tp2_resident_control.py is hashed into the identity)"
        )

    result: dict[str, object] = {
        "kind": "tp2_prefill_schedule_suite_rollup",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "diagnostic_only",
        "performance_claim": False,
        "production_qualified": False,
        "teacher": str(args.teacher),
        "teacher_sha256": teacher_sha,
        "thresholds": PRODUCTION_GATE,
        "large_kl_threshold": LARGE_KL,
        "steps": args.steps,
        "revisions": {arm: entry["artifact"]["source_revision"] for arm, entry in passes.items()},
        "host_process_diffs": {
            arm: sorted(entry["artifact"]["teacher_identity_diff"])
            for arm, entry in passes.items()
        },
        "arms": {},
        "prompts": {},
        "teacher_flatness": {},
    }

    suite_ids = list(teacher["suite"]["ids"])
    curves: dict[str, dict[str, np.ndarray]] = {}
    for arm, entry in passes.items():
        artifact = entry["artifact"]
        per_prompt = {
            prompt_id: np.asarray(
                prompt["comparisons"][f"teacher_vs_{arm}"]["kl_curve"], dtype=np.float64
            )
            for prompt_id, prompt in artifact["prompts"].items()
            if f"teacher_vs_{arm}" in prompt["comparisons"]
        }
        if not per_prompt:
            raise ValueError(f"{entry['path']} has no teacher_vs_{arm} comparisons")
        curves[arm] = per_prompt
        result["arms"][arm] = {
            "artifact": str(entry["path"]),
            "artifact_sha256": hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest(),
            "prefill_schedule": artifact.get("arms", {}).get(arm, {}).get("prefill_schedule"),
            "suite_aggregate": artifact["suite_aggregate"].get(arm, {}),
            **_authorized_horizon(per_prompt, args.horizon_floor),
            "first_breaching_depth": {
                prompt_id: _first_breaching_depth(curve, args.horizon_floor)
                for prompt_id, curve in per_prompt.items()
            },
        }

    # Per-prompt envelope table across arms.
    for prompt_id in suite_ids:
        row: dict[str, object] = {"suite_index": suite_ids.index(prompt_id), "arms": {}}
        for arm, per_prompt in curves.items():
            if prompt_id not in per_prompt:
                continue
            kl = per_prompt[prompt_id]
            row["arms"][arm] = {
                "mean_kl": float(kl.mean()),
                "p95_kl": float(np.percentile(kl, 95)),
                "p99_kl": float(np.percentile(kl, 99)),
                "max_kl": float(kl.max()),
                "positions_over_max_kl": int((kl > PRODUCTION_GATE["max_kl"]).sum()),
                "failures": _envelope_failures(kl),
                "first_breaching_depth": _first_breaching_depth(kl, args.horizon_floor),
            }
        result["prompts"][prompt_id] = row

    # Teacher flatness at the positions that breach, against the whole suite.
    all_top1: list[np.ndarray] = []
    all_gap: list[np.ndarray] = []
    over_top1: list[np.ndarray] = []
    over_gap: list[np.ndarray] = []
    for prompt_id, row in result["prompts"].items():
        index = suite_ids.index(prompt_id)
        reference = np.asarray(
            np.load(teacher["arrays"][index]["path"], mmap_mode="r")[: args.steps],
            dtype=np.float64,
        )
        shapes = np.array([_distribution_shape(reference[i]) for i in range(len(reference))])
        all_top1.append(shapes[:, 0])
        all_gap.append(shapes[:, 1])
        for arm, per_prompt in curves.items():
            if prompt_id not in per_prompt:
                continue
            mask = per_prompt[prompt_id] > LARGE_KL
            if mask.any():
                over_top1.append(shapes[mask, 0])
                over_gap.append(shapes[mask, 1])
    top1 = np.concatenate(all_top1)
    gap = np.concatenate(all_gap)
    result["teacher_flatness"] = {
        "rows": int(top1.size),
        "suite_top1_prob_median": float(np.median(top1)),
        "suite_top1_prob_p05": float(np.percentile(top1, 5)),
        "suite_top2_gap_median": float(np.median(gap)),
        "suite_rows_below_top1_prob_0.9": int((top1 < 0.9).sum()),
        "large_kl_rows": int(sum(part.size for part in over_top1)),
        "large_kl_top1_prob_median": (
            float(np.median(np.concatenate(over_top1))) if over_top1 else None
        ),
        "large_kl_top1_prob_max": (
            float(np.concatenate(over_top1).max()) if over_top1 else None
        ),
        "large_kl_top2_gap_median": (
            float(np.median(np.concatenate(over_gap))) if over_gap else None
        ),
        "note": (
            "flatness is necessary but not sufficient: most flat rows carry no "
            "large KL because the two implementations still agree there"
        ),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"artifact: {args.json}")
    for arm, record in result["arms"].items():
        print(
            f"  {arm:12s} schedule {record['prefill_schedule']:<12} "
            f"horizon {record['authorized_horizon']:3d}  "
            f"prompts breaching {record['suite_aggregate'].get('prompts_breaching_ceiling')}"
            f"/{record['suite_aggregate'].get('prompts')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
