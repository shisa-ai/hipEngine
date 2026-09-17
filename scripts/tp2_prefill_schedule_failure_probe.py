"""Compare prefill schedules at the sustained-gate failure point, one revision.

The sustained D128 gate scores the TP2 arm (token-serial prefill) against a
bulk-prefill TP1 teacher, and it fails on ``mixed_ja_en_translate`` at decode
index 82. The recorded localization showed the divergence is dominated by the
prefill *schedule* (two TP1 schedules differ by more than either does from TP2)
but never ran the candidate that would remove the schedule difference: the TP2
session's own rank-local bulk prefill.

This probe runs up to four arms over the same teacher-forced prefix on the same
revision and reports the whole per-position KL curve for each:

``tp1-bulk``    the teacher arm, re-run to confirm it reproduces its own rows
``tp1-serial``  the same TP1 session with token-serial prefill (schedule
                sensitivity reference: no TP2 arithmetic is involved)
``tp2-serial``  the TP2 session's committed prefill route
``tp2-bulk``    the TP2 rank-local bulk prefill candidate

Suite mode (``--prompts all``) walks the whole teacher suite instead of one
prompt and reports, per arm, how many prompts breach the ceiling and the
suite-wide envelope. Arms are processed one at a time, so only one session is
resident on the devices at once; ``--store-rows DIR`` writes each prompt's rows
for a later pass to compare against with ``--compare-rows DIR`` (this is how the
two TP2 schedules are compared across the suite without holding both sessions).

Diagnostics only: this makes no performance or product claim and promotes
nothing.

Usage::

    PYTHONPATH=$PWD python3 scripts/tp2_prefill_schedule_failure_probe.py \\
        --teacher /path/to/quality-tp1-d0.json \\
        --prompt-id mixed_ja_en_translate --steps 128 \\
        --json benchmarks/results/<artifact>.json

    PYTHONPATH=$PWD python3 scripts/tp2_prefill_schedule_failure_probe.py \\
        --teacher /path/to/quality-tp1-d0.json --prompts all --steps 128 \\
        --arms tp2-serial --store-rows /tmp/tp2-suite-rows \\
        --json /tmp/tp2-suite-serial.json
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

ARMS = ("tp1-bulk", "tp1-serial", "tp2-serial", "tp2-bulk")


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


def _compare(
    teacher_row: np.ndarray, candidate_row: np.ndarray, *, with_shape: bool
) -> dict[str, object]:
    """Envelope, curves and (optionally) logit detail for one arm-vs-reference."""

    kl, top1 = _kl_rows(np.asarray(teacher_row, dtype=np.float32), candidate_row)
    breaches = np.nonzero(kl > PRODUCTION_GATE["max_kl"])[0]
    worst = int(kl.argmax())
    summary: dict[str, object] = {
        **_envelope_summary(kl, top1),
        "max_kl_index": worst,
        "first_breach_index": int(breaches[0]) if breaches.size else None,
        "kl_curve": [round(float(value), 8) for value in kl],
    }
    if with_shape:
        delta = np.abs(
            np.asarray(candidate_row[worst], dtype=np.float64)
            - np.asarray(teacher_row[worst], dtype=np.float64)
        )
        summary["teacher_shape_at_worst"] = _distribution_shape(teacher_row[worst])
        summary["arm_shape_at_worst"] = _distribution_shape(candidate_row[worst])
        summary["max_abs_logit_diff_at_worst"] = float(delta.max())
        summary["mean_abs_logit_diff_at_worst"] = float(delta.mean())
    return summary


def _aggregate(per_prompt: dict[str, dict[str, object]]) -> dict[str, object]:
    """Suite-wide rollup of one arm's per-prompt comparisons."""

    if not per_prompt:
        return {}
    worst_prompt = max(per_prompt, key=lambda name: per_prompt[name]["max_kl"])
    return {
        "prompts": len(per_prompt),
        "prompts_breaching_ceiling": sum(
            1 for summary in per_prompt.values() if summary["positions_over_max_kl"]
        ),
        "positions_over_max_kl": sum(
            summary["positions_over_max_kl"] for summary in per_prompt.values()
        ),
        "positions_total": sum(summary["rows"] for summary in per_prompt.values()),
        "mean_kl_over_prompts": float(
            np.mean([summary["mean_kl"] for summary in per_prompt.values()])
        ),
        "worst_prompt_mean_kl": max(
            summary["mean_kl"] for summary in per_prompt.values()
        ),
        "mean_p95_kl_over_prompts": float(
            np.mean([summary["p95_kl"] for summary in per_prompt.values()])
        ),
        "worst_p95_kl": max(summary["p95_kl"] for summary in per_prompt.values()),
        "worst_max_kl": per_prompt[worst_prompt]["max_kl"],
        "worst_max_kl_prompt": worst_prompt,
        "min_top1_agreement": min(
            summary["top1_agreement"] for summary in per_prompt.values()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prompt-id", default="mixed_ja_en_translate")
    parser.add_argument(
        "--prompts",
        default=None,
        help="'all' for the whole teacher suite, or a comma-separated list of "
        "prompt ids; defaults to --prompt-id",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated subset of " + ",".join(ARMS),
    )
    parser.add_argument(
        "--store-rows",
        type=Path,
        default=None,
        help="directory to write each prompt's logit rows into, for a later "
        "pass to compare against with --compare-rows",
    )
    parser.add_argument(
        "--compare-rows",
        type=Path,
        default=None,
        help="directory written by --store-rows; every arm found in it for a "
        "prompt is compared against this pass's rows for the same prompt",
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

    suite_ids = list(teacher["suite"]["ids"])
    if args.prompts in (None, ""):
        prompt_ids = [args.prompt_id]
    elif args.prompts == "all":
        prompt_ids = suite_ids
    else:
        prompt_ids = [name.strip() for name in args.prompts.split(",") if name.strip()]
    unknown = [name for name in prompt_ids if name not in suite_ids]
    if unknown:
        raise ValueError(f"prompts not in the teacher suite: {unknown}")

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown_arms = set(arms) - set(ARMS)
    if unknown_arms:
        raise ValueError(f"unknown arms: {sorted(unknown_arms)}")
    if args.store_rows is not None:
        args.store_rows.mkdir(parents=True, exist_ok=True)

    single = len(prompt_ids) == 1
    failure_index = 82 if args.steps > 82 else args.steps - 1
    result: dict[str, object] = {
        "kind": "tp2_prefill_schedule_failure_probe",
        "schema": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "diagnostic_only",
        "performance_claim": False,
        "production_qualified": False,
        "model": str(args.model),
        "model_sha256": identity["model_sha256"],
        "source_revision": identity["source_revision"],
        "source_sha256": identity["source_sha256"],
        "profile": profile,
        "teacher": str(args.teacher),
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        "teacher_identity_diff": diff,
        "host_process_drift_allowed": bool(args.allow_host_process_drift),
        "arms_requested": arms,
        "prompt_ids": prompt_ids,
        "steps": args.steps,
        "failure_index": failure_index,
        "thresholds": PRODUCTION_GATE,
        "host": {
            "node": platform.node(),
            "nice": os.nice(0),
            "argv": sys.argv[1:],
        },
        "prompts": {},
        "suite_aggregate": {},
    }

    # Per arm: one session, then every prompt in order. The session is reset by
    # each prefill, so one build covers the whole suite; holding two sessions of
    # different arms at once would double the resident footprint per rank.
    kept_rows: dict[str, np.ndarray] = {}
    kept_reference: dict[str, np.ndarray] = {}
    for arm in arms:
        if arm.startswith("tp1"):
            adapter = create_native_adapter(args.model, "tp1-d0", capacity=200)
            if arm == "tp1-serial":
                adapter.prefill_use_bulk = False
        else:
            adapter = create_native_adapter(
                args.model, "tp2", capacity=200, bulk_prefill=arm == "tp2-bulk"
            )
        arm_record: dict[str, object] = {
            "prefill_schedule": adapter.prefill_schedule,
            "prefill_use_bulk": adapter.prefill_use_bulk,
            "seconds": 0.0,
        }
        result["arms"] = {**(result.get("arms") or {}), arm: arm_record}
        started = time.perf_counter()
        try:
            adapter.prepare()
            for prompt_id in prompt_ids:
                index = suite_ids.index(prompt_id)
                tokens = teacher["suite"]["tokens"][index]
                forced = teacher["forced_inputs"][index][: args.steps]
                reference = np.asarray(
                    np.load(teacher["arrays"][index]["path"], mmap_mode="r")[: args.steps],
                    dtype=np.float32,
                )
                prompt = result["prompts"].setdefault(
                    prompt_id,
                    {
                        "suite_index": index,
                        "prompt_tokens": len(tokens),
                        "positions": [len(tokens) + i for i in range(args.steps)],
                        "arms": {},
                        "comparisons": {},
                    },
                )
                logits, inputs, _controls = sustained_trajectory(
                    adapter, tokens, forced=forced, steps=args.steps
                )
                prompt["arms"][arm] = {
                    "prefill_schedule": adapter.prefill_schedule,
                    "prefill_use_bulk": adapter.prefill_use_bulk,
                    "last_row_sha256": _row_hash(logits[-1:]),
                    "forced_inputs_match": list(inputs) == list(forced),
                    "finite": bool(np.isfinite(logits).all()),
                }
                prompt["comparisons"][f"teacher_vs_{arm}"] = _compare(
                    reference, logits, with_shape=True
                )
                if single:
                    # One prompt: keep the rows so the artifact can carry them.
                    # Suite mode drops them per prompt to stay inside memory.
                    kept_rows[arm] = logits
                    kept_reference[prompt_id] = reference
                if args.store_rows is not None:
                    np.save(args.store_rows / f"{arm}-{prompt_id}.npy", logits)
                if args.compare_rows is not None:
                    for path in sorted(args.compare_rows.glob(f"*-{prompt_id}.npy")):
                        other = path.name[: -len(f"-{prompt_id}.npy")]
                        if other == arm or other not in ARMS:
                            continue
                        stored = np.load(path)
                        if stored.shape != logits.shape:
                            raise ValueError(
                                f"stored {other} rows for {prompt_id} have shape "
                                f"{stored.shape}, not {logits.shape}"
                            )
                        prompt["comparisons"][f"{other}_vs_{arm}"] = _compare(
                            stored, logits, with_shape=False
                        )
        except Exception as error:  # noqa: BLE001 - record the arm's own failure
            arm_record["error"] = f"{type(error).__name__}: {error}"
        finally:
            arm_record["seconds"] = round(time.perf_counter() - started, 1)
            try:
                adapter.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass

    # Suite rollup per arm, over the prompts that produced comparisons.
    for arm in arms:
        per_prompt = {
            prompt_id: prompt["comparisons"][f"teacher_vs_{arm}"]
            for prompt_id, prompt in result["prompts"].items()
            if f"teacher_vs_{arm}" in prompt["comparisons"]
        }
        if per_prompt:
            result["suite_aggregate"][arm] = _aggregate(per_prompt)
    for other, target in (("tp1-serial", "tp2-serial"), ("tp2-serial", "tp2-bulk")):
        name = f"{other}_vs_{target}"
        per_prompt = {
            prompt_id: prompt["comparisons"][name]
            for prompt_id, prompt in result["prompts"].items()
            if name in prompt["comparisons"]
        }
        if per_prompt:
            result["suite_aggregate"][name] = _aggregate(per_prompt)

    if single:
        # Keep the one-prompt artifact shape the earlier evidence used: a flat
        # comparisons map plus the original failure index.
        prompt_id = prompt_ids[0]
        prompt = result["prompts"][prompt_id]
        result["prompt_id"] = prompt_id
        result["tokens"] = list(teacher["suite"]["tokens"][suite_ids.index(prompt_id)])
        result["positions"] = prompt["positions"]
        result["comparisons"] = prompt["comparisons"]
        reference = kept_reference[prompt_id]
        result["failure_position"] = int(prompt["positions"][failure_index])
        result["at_failure_index"] = {
            name: {
                "max_kl": summary["max_kl"],
                "mean_kl": summary["mean_kl"],
                "p99_kl": summary["p99_kl"],
                "kl_at_index": summary["kl_curve"][failure_index],
                "first_breach_index": summary.get("first_breach_index"),
                "positions_over_max_kl": summary["positions_over_max_kl"],
            }
            for name, summary in prompt["comparisons"].items()
        }
        result["teacher_shape_at_failure"] = _distribution_shape(reference[failure_index])
        result["teacher_top16_at_failure"] = [
            {"token": int(token), "logit": float(value)}
            for token, value in sorted(
                (
                    (int(token), float(reference[failure_index][token]))
                    for token in np.argsort(reference[failure_index])[-16:]
                ),
                key=lambda item: -item[1],
            )
        ]

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(f"artifact: {args.json}")
        if single:
            # Full logit rows at each comparison's worst position, so a follow-up
            # question about this failure does not need another GPU run. Keys are
            # '<arm>_at_<index>'; 248320 x 4 bytes each.
            prompt = result["prompts"][prompt_ids[0]]
            reference = kept_reference[prompt_ids[0]]
            worst_rows: dict[str, np.ndarray] = {}
            for name, summary in prompt["comparisons"].items():
                index = summary["max_kl_index"]
                if name.startswith("teacher_vs_"):
                    worst_rows[f"teacher_at_{index}"] = reference[index]
                    arm_name = name.replace("teacher_vs_", "")
                    if arm_name in kept_rows:
                        worst_rows[f"{arm_name}_at_{index}"] = kept_rows[arm_name][index]
            rows_path = args.json.with_suffix(".worst-rows.npz")
            np.savez(rows_path, **worst_rows)
            result["worst_rows_path"] = str(rows_path)
            result["worst_rows_note"] = (
                "full logit rows at each comparison's worst position; keys are "
                "'<arm>_at_<index>' and 'teacher_at_<index>'"
            )
            args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "suite_aggregate": result["suite_aggregate"],
                "arms": result.get("arms"),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
