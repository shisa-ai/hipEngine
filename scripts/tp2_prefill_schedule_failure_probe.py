"""Compare prefill schedules at the sustained-gate failure point, one revision.

The sustained D128 gate scores the TP2 arm (token-serial prefill) against a
bulk-prefill TP1 teacher, and it fails on ``mixed_ja_en_translate`` at decode
index 82. The recorded localization showed the divergence is dominated by the
prefill *schedule* (two TP1 schedules differ by more than either does from TP2)
but never ran the candidate that would remove the schedule difference: the TP2
session's own rank-local bulk prefill.

This probe runs four arms over the same teacher-forced prefix on the same
revision and reports the whole per-position KL curve for each:

``tp1-bulk``    the teacher arm, re-run to confirm it reproduces its own rows
``tp1-serial``  the same TP1 session with token-serial prefill (schedule
                sensitivity reference: no TP2 arithmetic is involved)
``tp2-serial``  the TP2 session's committed prefill route
``tp2-bulk``    the TP2 rank-local bulk prefill candidate

Diagnostics only: this is one prompt at the failing horizon, not the suite gate,
and it makes no performance or product claim.

Usage::

    PYTHONPATH=$PWD python3 scripts/tp2_prefill_schedule_failure_probe.py \\
        --teacher /path/to/quality-tp1-d0.json \\
        --prompt-id mixed_ja_en_translate --steps 128 \\
        --json benchmarks/results/<artifact>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tp2_resident_control import (  # noqa: E402
    bind_resident_profile,
    create_native_adapter,
)
from scripts.tp2_teacher_coverage_broad import (  # noqa: E402
    PRODUCTION_GATE,
    _kl_rows,
    _row_hash,
    sustained_trajectory,
)

#: Identity fields that describe the host process rather than the arithmetic.
#: The harness's identity guard compares them too; this probe reports a mismatch
#: in them instead of refusing, because this environment re-nices long-running
#: shells (``os.nice(0)`` is 16 for a process started immediately and -4 a few
#: seconds later), which would otherwise block every cross-arm comparison.
_HOST_PROCESS_FIELDS = ("host",)


def _identity_diff(teacher: dict, current: dict) -> dict[str, dict[str, str]]:
    diff: dict[str, dict[str, str]] = {}
    for key in teacher:
        if teacher[key] != current.get(key):
            diff[key] = {
                "teacher": json.dumps(teacher[key], sort_keys=True)[:400],
                "current": json.dumps(current.get(key), sort_keys=True)[:400],
            }
    return diff


def _distribution_shape(row: np.ndarray) -> dict[str, float]:
    """Shape of one logit row: how flat the distribution is at its top.

    A large KL at a position where the teacher's top-2 logits are nearly tied
    is a *near-tie* artifact: a tiny arithmetic difference moves probability
    mass between two near-equal candidates. A large KL on a sharp distribution
    is a real distributional difference. These two must not be read the same
    way, so every comparison records the teacher's shape at its worst position.
    """

    x = np.asarray(row, dtype=np.float64)
    order = np.sort(x)
    shifted = x - x.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return {
        "top1": int(np.argmax(x)),
        "top1_prob": float(probabilities.max()),
        "top2_logit_gap": float(order[-1] - order[-2]),
        "entropy": float(-(probabilities * np.log(probabilities + 1e-45)).sum()),
    }


def _envelope_summary(kl: np.ndarray, top1: np.ndarray) -> dict[str, float]:
    """The production envelope's own statistics for one comparison."""

    return {
        "rows": int(kl.size),
        "mean_kl": float(kl.mean()),
        "p95_kl": float(np.percentile(kl, 95)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(kl.max()),
        "top1_agreement": float(top1.mean()),
        "positions_over_0.01": int((kl > 0.01).sum()),
        "positions_over_max_kl": int((kl > PRODUCTION_GATE["max_kl"]).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prompt-id", default="mixed_ja_en_translate")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--arms",
        default="tp1-bulk,tp1-serial,tp2-serial,tp2-bulk",
        help="comma-separated subset of tp1-bulk,tp1-serial,tp2-serial,tp2-bulk",
    )
    parser.add_argument(
        "--allow-host-process-drift",
        action="store_true",
        help="proceed when the only identity differences are host-process fields "
        "(re-nicing), and record the diff in the artifact",
    )
    args = parser.parse_args(argv)

    teacher = json.loads(args.teacher.read_text())
    if teacher.get("status") != "complete":
        raise ValueError("teacher capture is not complete")
    profile = bind_resident_profile("production")
    from scripts.tp2_matched_ar_baseline import product_identity

    identity = product_identity(args.model)
    diff = _identity_diff(teacher["identity"], identity)
    blocking = {k: v for k, v in diff.items() if k not in _HOST_PROCESS_FIELDS}
    if blocking:
        raise ValueError(f"teacher identity mismatch outside host fields: {sorted(blocking)}")
    if diff and not args.allow_host_process_drift:
        raise ValueError(
            "host-process identity drift detected; pass --allow-host-process-drift "
            f"to proceed and record it: {sorted(diff)}"
        )

    prompt_id = args.prompt_id
    if prompt_id not in teacher["suite"]["ids"]:
        raise ValueError(f"prompt {prompt_id!r} is not in the teacher suite")
    index = teacher["suite"]["ids"].index(prompt_id)
    tokens = teacher["suite"]["tokens"][index]
    forced = teacher["forced_inputs"][index][: args.steps]
    reference = np.load(teacher["arrays"][index]["path"], mmap_mode="r")[: args.steps]

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = set(arms) - {"tp1-bulk", "tp1-serial", "tp2-serial", "tp2-bulk"}
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")

    result: dict[str, object] = {
        "kind": "tp2_prefill_schedule_failure_probe",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "diagnostic_only",
        "performance_claim": False,
        "production_qualified": False,
        "model": str(args.model),
        "model_sha256": identity["model_sha256"],
        "source_revision": identity["source_revision"],
        "profile": profile,
        "teacher": str(args.teacher),
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        "teacher_identity_diff": diff,
        "host_process_drift_allowed": bool(args.allow_host_process_drift),
        "prompt_id": prompt_id,
        "tokens": list(tokens),
        "positions": [len(tokens) + i for i in range(args.steps)],
        "steps": args.steps,
        "thresholds": PRODUCTION_GATE,
        "host": {
            "node": platform.node(),
            "nice": os.nice(0),
            "argv": sys.argv[1:],
        },
        "arms": {},
    }

    rows: dict[str, np.ndarray] = {}
    for arm in arms:
        if arm.startswith("tp1"):
            adapter = create_native_adapter(args.model, "tp1-d0", capacity=200)
            if arm == "tp1-serial":
                adapter.prefill_use_bulk = False
        else:
            adapter = create_native_adapter(
                args.model, "tp2", capacity=200, bulk_prefill=arm == "tp2-bulk"
            )
        started = time.perf_counter()
        try:
            adapter.prepare()
            logits, inputs, _controls = sustained_trajectory(
                adapter, tokens, forced=forced, steps=args.steps
            )
            rows[arm] = logits
            elapsed = time.perf_counter() - started
            result["arms"][arm] = {
                "mode": adapter.owner.mode if hasattr(adapter.owner, "mode") else "tp2",
                "prefill_schedule": adapter.prefill_schedule,
                "seconds": round(elapsed, 1),
                "last_row_sha256": _row_hash(logits[-1:]),
                "forced_inputs_match": list(inputs) == list(forced),
                "finite": bool(np.isfinite(logits).all()),
            }
        except Exception as error:  # noqa: BLE001 - record the arm's own failure
            result["arms"][arm] = {
                "prefill_schedule": adapter.prefill_schedule,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass

    # Per-position KL against the teacher for every arm that produced rows, plus
    # the schedule-sensitivity reference pairs among the arms themselves.
    comparisons: dict[str, object] = {}
    teacher_reference = np.asarray(reference, dtype=np.float32)
    worst_rows: dict[str, np.ndarray] = {}
    for arm, logits in rows.items():
        kl, top1 = _kl_rows(teacher_reference, logits)
        breaches = np.nonzero(kl > PRODUCTION_GATE["max_kl"])[0]
        worst = int(kl.argmax())
        delta = np.abs(
            logits[worst].astype(np.float64) - teacher_reference[worst].astype(np.float64)
        )
        # One row per array, not the whole reference: a full horizon of logits is
        # 127 MB, which is not a compact artifact.
        worst_rows[f"teacher_at_{worst}"] = teacher_reference[worst]
        worst_rows[f"{arm}_at_{worst}"] = logits[worst]
        comparisons[f"teacher_vs_{arm}"] = {
            **_envelope_summary(kl, top1),
            "max_kl_index": worst,
            "max_kl_position": int(result["positions"][worst]),
            "first_breach_index": int(breaches[0]) if breaches.size else None,
            "first_breach_position": (
                int(result["positions"][int(breaches[0])]) if breaches.size else None
            ),
            "teacher_shape_at_worst": _distribution_shape(teacher_reference[worst]),
            "arm_shape_at_worst": _distribution_shape(logits[worst]),
            "max_abs_logit_diff_at_worst": float(delta.max()),
            "mean_abs_logit_diff_at_worst": float(delta.mean()),
            "kl_curve": [round(float(value), 8) for value in kl],
        }
    for left, right in (
        ("tp1-bulk", "tp1-serial"),
        ("tp1-serial", "tp2-serial"),
        ("tp1-bulk", "tp2-serial"),
        ("tp1-bulk", "tp2-bulk"),
        ("tp2-serial", "tp2-bulk"),
    ):
        if left not in rows or right not in rows:
            continue
        kl, top1 = _kl_rows(rows[left], rows[right])
        comparisons[f"{left}_vs_{right}"] = {
            **_envelope_summary(kl, top1),
            "max_kl_index": int(kl.argmax()),
            "kl_curve": [round(float(value), 8) for value in kl],
        }
    result["comparisons"] = comparisons

    # The row this probe exists to explain: the original gate failure.
    failure_index = 82 if args.steps > 82 else args.steps - 1
    result["failure_index"] = failure_index
    result["failure_position"] = int(result["positions"][failure_index])
    result["at_failure_index"] = {
        name: {
            "max_kl": summary["max_kl"],
            "mean_kl": summary["mean_kl"],
            "p99_kl": summary["p99_kl"],
            "kl_at_index": summary["kl_curve"][failure_index],
            "first_breach_index": summary.get("first_breach_index"),
            "positions_over_max_kl": summary["positions_over_max_kl"],
        }
        for name, summary in comparisons.items()
    }
    result["teacher_shape_at_failure"] = _distribution_shape(
        teacher_reference[failure_index]
    )
    # Full logit rows at each comparison's worst position, so a follow-up
    # question about this failure does not need another GPU run. Keys are
    # '<arm>_at_<index>'; 248320 x 4 bytes each.
    if args.json is not None and worst_rows:
        rows_path = args.json.with_suffix(".worst-rows.npz")
        np.savez(rows_path, **worst_rows)
        result["worst_rows_path"] = str(rows_path)
        result["worst_rows_note"] = (
            "full logit rows at each comparison's worst position; keys are "
            "'<arm>_at_<index>' and 'teacher_at_<index>'"
        )
        result["teacher_top16_at_failure"] = [
            {"token": int(token), "logit": float(value)}
            for token, value in sorted(
                (
                    (int(token), float(teacher_reference[failure_index][token]))
                    for token in np.argsort(teacher_reference[failure_index])[-16:]
                ),
                key=lambda item: -item[1],
            )
        ]

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(f"artifact: {args.json}")
    print(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "steps": args.steps,
                "at_failure_index": result["at_failure_index"],
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
