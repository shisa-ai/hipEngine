#!/usr/bin/env python3
"""Matched TP1-vs-TP2 true-AR protocol harness (natural-length diagnostic cell).

Status: this is a **diagnostic protocol**, not a qualified baseline. It measures
the optimized TP1 resident graph-replay route on each physical GPU against the
TP2 graphed session and validates that the two arms did the same work under the
same accounting. It does not by itself qualify a TP2 result: numerical,
category, determinism, reset, and provenance gates are separate and are all
recorded as unmeasured here, which makes ``qualified`` false.

Declared diagnostic cell ``C1-natural`` (predeclared; a run that cannot satisfy
it is recorded as blocked, never silently relaxed):

* **Context is natural, not a fixed 128.** Each prompt is tokenized with
  ``build_chat_prompt`` and the per-prompt context is that prompt's own length.
  There is no fixed-length construction, so these are *not* context-128
  measurements and must not be labelled as such. A fixed-length cell requires a
  declared deterministic token extension/truncation transform first.
* output = 128 timed decode transitions, warmup = 0, greedy, no EOS stop.
* Full-vocabulary logits transferred on every timed decode step on both arms.
  This is a **matched-transfer diagnostic only**: it is deliberately not the
  optimized TP1 product baseline (whose decode loop transfers logits only on the
  last step), so its TP1 rate is a lower bound and must not be published as the
  optimized TP1 throughput.
* TP1 graph replay and the TP2 graphed schedule; graph capture and destruction
  are measured and reported separately.
* The headline window is an externally measured whole-generation wall (prefill +
  decode + capture + destruction) on both arms, so Python per-step argmax/list
  overhead that TP2 carries internally is included on both sides. Adjusted
  (capture- and destruction-free) windows must be positive and finite.
* Sampled outputs are aligned explicitly, not by count: both arms produce 128
  sampled decode outputs, with the prefill sample recorded separately.

Usage (orchestrator):
    python3 scripts/tp2_matched_ar_baseline.py --run --reps 3 --json out.json
Worker (invoked by the orchestrator):
    python3 scripts/tp2_matched_ar_baseline.py --arm tp2 --worker-reps 3 --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_category_bench import BenchError, load_prompt_rows  # noqa: E402
from scripts.gguf_true_ar_category_bench import (  # noqa: E402
    DEFAULT_PROMPTS,
    build_chat_prompt,
)
from scripts.tp2_teacher_coverage_broad import (  # noqa: E402
    _device_identities,
    _git_dirty,
    _git_revision,
    _host_identity,
    _model_sha256,
    _resolved_route,
)
#: The dense model this cell measures; the inherited MTP-suite default is MoE,
#: which the dense MLP shard plan correctly refuses.
DEFAULT_DENSE_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_HELDOUT_PROMPTS = (
    REPO_ROOT / "benchmarks" / "prompts" / "laguna-target-ar-code-general-ja-heldout.jsonl"
)

#: Predeclared diagnostic cell. Every field is normative.
CELL_C1_NATURAL: dict[str, Any] = {
    "name": "C1-natural",
    "context": "natural",
    "output_tokens": 128,
    "warmup_decode_tokens": 0,
    "sampling": "greedy",
    "eos": None,
    "logits_per_decode_step": True,
    "graph_replay_decode": True,
    "tp1_product_baseline": False,
}

TP1_ARMS = ("tp1-d0", "tp1-d1")
ARMS = ("tp1-d0", "tp1-d1", "tp2")
QUALIFICATION_GATES = (
    "numeric_logits_measured",
    "category_measured",
    "determinism_measured",
    "reset_measured",
    "provenance_complete",
)


class MatchedBaselineError(RuntimeError):
    """A declared-cell violation that must fail rather than produce a ratio."""


# -- prompt suite -----------------------------------------------------------


def load_cell_prompts(canonical: Path, heldout: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Canonical suite plus heldout-only rows, deduplicated by id.

    A missing or empty heldout file is a failure, never a silent skip.
    """

    failures: list[str] = []
    if heldout is None or not heldout.exists():
        failures.append(f"heldout suite missing: {heldout}")
        return [], failures
    try:
        rows = load_prompt_rows(canonical)
        heldout_rows = load_prompt_rows(heldout)
    except BenchError as error:
        return [], [f"prompt load failed: {error}"]
    seen = {str(row["id"]) for row in rows}
    for row in heldout_rows:
        if str(row["id"]) in seen:
            continue
        rows.append(row)
        seen.add(str(row["id"]))
    if not heldout_rows:
        failures.append(f"heldout suite has no rows: {heldout}")
    heldout_ids = {str(r["id"]) for r in heldout_rows} - {
        str(r["id"]) for r in load_prompt_rows(canonical)
    }
    if not heldout_ids:
        failures.append("no heldout-only rows after deduplication")
    return rows, failures


def write_prompt_file(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


# -- accounting -------------------------------------------------------------


def _ids_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode("ascii")).hexdigest()


def validate_accounting(arm: str, row: dict[str, Any], *, cell: dict[str, Any]) -> list[str]:
    """Per-prompt protocol/accounting failures. Pure and fail-closed."""

    failures: list[str] = []
    prompt_id = str(row.get("id", "?"))
    output = int(cell["output_tokens"])

    def need(condition: bool, message: str) -> None:
        if not condition:
            failures.append(f"{prompt_id}: {message}")

    # -- declared work ------------------------------------------------------
    need(int(row.get("output_tokens", -1)) == output, f"output_tokens != {output}")
    need(
        int(row.get("timed_decode_transitions", -1)) == output,
        f"timed_decode_transitions != {output}",
    )
    # -- context semantics --------------------------------------------------
    context = cell["context"]
    if context == "natural":
        need(
            int(row.get("context_tokens", -1)) == int(row.get("prompt_tokens", -2)),
            "natural-length context_tokens != prompt_tokens",
        )
    else:
        need(
            int(row.get("prompt_tokens", -1)) == int(context),
            f"prompt_tokens != declared context {context}",
        )
        need(False, "fixed-length cell construction is not implemented")
    need(int(row.get("prompt_tokens", 0)) > 0, "prompt_tokens must be positive")
    need(
        int(row.get("context_position_at_timing_start", -2))
        == int(row.get("prompt_tokens", -1)) + int(cell["warmup_decode_tokens"]),
        "context_position_at_timing_start != prompt_tokens + warmup",
    )
    # -- route --------------------------------------------------------------
    need(bool(row.get("graph_effective", False)), "graph replay not effective")
    need(
        bool(row.get("logits_per_decode_step", False))
        == bool(cell["logits_per_decode_step"]),
        "logits_per_decode_step does not match the cell",
    )
    # -- logits -------------------------------------------------------------
    need(bool(row.get("finite_final_logits", False)), "final logits not finite")
    need(
        bool(row.get("finite_all_decode_logits", False)),
        "all decode-step logits not verified finite",
    )
    need(str(row.get("eos_policy", "")) == "none", "eos_policy != 'none'")
    # -- sampled-output alignment (not counts alone) ------------------------
    sampled = list(row.get("sampled_output_ids", []))
    need(len(sampled) == output, f"sampled_output_ids {len(sampled)} != {output}")
    need(row.get("sampled_output_sha256") == _ids_hash(sampled), "sampled_output_sha256 mismatch")
    need(row.get("prefill_sample_id") is not None, "prefill_sample_id missing")
    # -- timing windows -----------------------------------------------------
    for key in ("total_generation_ms", "capture_ms", "destroy_ms"):
        value = row.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            failures.append(f"{prompt_id}: {key} must be finite and non-negative")
    total = float(row.get("total_generation_ms", 0.0))
    adjusted = total - float(row.get("capture_ms", 0.0)) - float(row.get("destroy_ms", 0.0))
    need(math.isfinite(adjusted) and adjusted > 0.0, "adjusted window must be positive and finite")
    need(
        float(row.get("prefill_ms", -1.0)) >= 0.0,
        "prefill_ms must be non-negative",
    )
    return failures


def validate_rep(
    rep: dict[str, Any],
    *,
    required_arms: Sequence[str] = ARMS,
    seen_run_ids: dict[str, str] | None = None,
) -> list[str]:
    """Rep completeness and execution-identity reuse. Pure and fail-closed."""

    failures: list[str] = []
    index = rep.get("rep")
    present = rep.get("arms", {})
    missing = [arm for arm in required_arms if arm not in present]
    if missing:
        failures.append(f"rep {index}: missing arms {missing}")
    run_ids = rep.get("run_ids", {})
    for arm in required_arms:
        run_id = run_ids.get(arm)
        if not run_id:
            failures.append(f"rep {index}: {arm} has no execution run id")
            continue
        if seen_run_ids is not None:
            if run_id in seen_run_ids:
                failures.append(
                    f"rep {index}: {arm} reused execution run id {run_id} "
                    f"(first seen in rep {seen_run_ids[run_id]})"
                )
            else:
                seen_run_ids[run_id] = str(index)
    if not rep.get("independent", False):
        failures.append(f"rep {index}: not marked as an independent execution")
    return failures


# -- aggregation and ratios -------------------------------------------------


def adjusted_ms(row: dict[str, Any]) -> float:
    return (
        float(row["total_generation_ms"])
        - float(row.get("capture_ms", 0.0))
        - float(row.get("destroy_ms", 0.0))
    )


def aggregate_rows(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    output_tokens = sum(int(r["output_tokens"]) for r in rows)
    adjusted = sum(adjusted_ms(r) for r in rows)
    return {
        "prompts": len(rows),
        "total_output_tokens": output_tokens,
        "adjusted_ms": adjusted,
        "total_generation_ms": sum(float(r["total_generation_ms"]) for r in rows),
        "capture_ms": sum(float(r.get("capture_ms", 0.0)) for r in rows),
        "destroy_ms": sum(float(r.get("destroy_ms", 0.0)) for r in rows),
        "adjusted_tok_s": 1000.0 * output_tokens / adjusted if adjusted > 0 else 0.0,
    }


def category_rows(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)
    return {cat: aggregate_rows(group, arm=arm) for cat, group in sorted(grouped.items())}


def compute_ratios(reps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """TP2/faster-TP1 ratio per rep, only when every rep is complete.

    Returns ``None`` when the ratio cannot be computed; callers must not report
    a ratio for an incomplete or failing run.
    """

    per_rep: list[dict[str, Any]] = []
    seen_run_ids: dict[str, int] = {}
    for rep in reps:
        arms = rep.get("arms", {})
        if not all(arm in arms for arm in ARMS):
            return None
        if not rep.get("complete", False):
            return None
        for run_id in rep.get("run_ids", {}).values():
            if run_id in seen_run_ids:
                return None
            seen_run_ids[run_id] = int(rep["rep"])
        tp2 = float(arms["tp2"]["adjusted_tok_s"])
        tp1 = {arm: float(arms[arm]["adjusted_tok_s"]) for arm in TP1_ARMS}
        faster_arm = max(tp1, key=lambda a: tp1[a])
        faster = tp1[faster_arm]
        if not (faster > 0 and tp2 > 0):
            return None
        per_rep.append(
            {
                "rep": int(rep["rep"]),
                "tp2_tok_s": tp2,
                "tp1_tok_s": tp1,
                "faster_tp1_arm": faster_arm,
                "faster_tp1_tok_s": faster,
                "tp2_vs_faster_tp1": tp2 / faster,
            }
        )
    ratios = [r["tp2_vs_faster_tp1"] for r in per_rep]
    if not ratios:
        return None
    return {
        "per_rep": per_rep,
        "reps": len(per_rep),
        "median_tp2_vs_faster_tp1": sorted(ratios)[len(ratios) // 2],
        "min_tp2_vs_faster_tp1": min(ratios),
        "max_tp2_vs_faster_tp1": max(ratios),
    }


# -- TP2 worker -------------------------------------------------------------


def run_tp2_arm(
    *, model: Path, prompts: Path, cell: dict[str, Any], reps: int = 1
) -> dict[str, Any]:
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    rows = load_prompt_rows(prompts)
    info = scan_gguf(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    token_rows = [build_chat_prompt(tokenizer, str(r["prompt"])) for r in rows]
    max_prompt = max(len(t) for t in token_rows)
    session = MlpTP2GenerationSession(
        model,
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=max_prompt + int(cell["output_tokens"]) + 1,
    )
    session_capture_ms = 0.0
    sweeps: list[list[dict[str, Any]]] = []
    try:
        # Capture is a one-time session cost; measure it explicitly rather than
        # reporting zero as if it did not happen.
        capture_start = time.perf_counter()
        session._ensure_graph_schedule()  # noqa: SLF001
        session_capture_ms = 1000.0 * (time.perf_counter() - capture_start)
        for _ in range(int(reps)):
            metrics: list[dict[str, Any]] = []
            for prompt_row, tokens in zip(rows, token_rows, strict=True):
                start = time.perf_counter()
                result = session.generate(
                    tokens,
                    max_new_tokens=int(cell["output_tokens"]),
                    eos_token_id=None,
                    capture_logits=True,
                )
                total_ms = 1000.0 * (time.perf_counter() - start)
                decode_traces = [t for t in result.step_traces if t.kind == "decode"]
                logits = result.logits
                # TP2 appends the input token before the forward, so token_ids[0]
                # is the prefill sample and token_ids[1:] are outputs 0..126; the
                # final sampled output is argmax of the last decode logits.
                sampled = [int(t) for t in result.token_ids[1:]]
                if logits is not None and len(logits):
                    sampled.append(int(np.argmax(logits[-1])))
                metrics.append(
                    {
                        "id": str(prompt_row["id"]),
                        "category": str(prompt_row["category"]),
                        "prompt_tokens": len(tokens),
                        "context_tokens": len(tokens),
                        "output_tokens": len(sampled),
                        "timed_decode_transitions": len(decode_traces),
                        "context_position_at_timing_start": len(tokens),
                        "graph_effective": str(session.schedule) == "graphed",
                        "logits_per_decode_step": True,
                        "eos_policy": "none",
                        "finite_final_logits": bool(
                            logits is not None and len(logits) and np.isfinite(logits[-1]).all()
                        ),
                        "finite_all_decode_logits": bool(
                            logits is not None and len(logits) and np.isfinite(logits).all()
                        ),
                        "prefill_sample_id": int(result.token_ids[0]),
                        "sampled_output_ids": sampled,
                        "sampled_output_sha256": _ids_hash(sampled),
                        "total_generation_ms": total_ms,
                        "prefill_ms": 1000.0
                        * sum(t.total_s for t in result.step_traces if t.kind == "prefill"),
                        "capture_ms": 0.0,
                        "destroy_ms": 0.0,
                    }
                )
            sweeps.append(metrics)
    finally:
        session.close()
    return {
        "arm": "tp2",
        "session_capture_ms": session_capture_ms,
        "sweeps": sweeps,
        "prompt_metrics": sweeps[0] if sweeps else [],
        "devices": _device_identities(session),
        "route": _resolved_route(session),
    }


# -- orchestrator -----------------------------------------------------------


def _tp1_command(*, model: Path, prompts: Path, output: Path, cell: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "gguf_true_ar_category_bench.py"),
        "--model",
        str(model),
        "--prompts",
        str(prompts),
        "--decode-tokens",
        str(cell["output_tokens"]),
        "--warmup-decode-tokens",
        str(cell["warmup_decode_tokens"]),
        "--graph-replay-decode",
        "--logits-every-decode-step",
        "--output",
        str(output),
    ]


def run_arm(
    arm: str,
    *,
    model: Path,
    prompts: Path,
    cell: dict[str, Any],
    workdir: Path,
    reps: int = 1,
    timeout_s: float = 2400.0,
) -> dict[str, Any]:
    """One independent execution of one arm, with a fresh run id."""

    run_id = uuid.uuid4().hex
    output = workdir / f"{arm}-{run_id}.json"
    if arm == "tp2":
        # TP2 runs in its own process: a resident TP2 session and a resident TP1
        # session must not hold contexts concurrently (observed GPU fault).
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tp2_matched_ar_baseline.py"),
            "--arm",
            "tp2",
            "--model",
            str(model),
            "--prompts",
            str(prompts),
            "--worker-reps",
            str(reps),
            "--json",
            str(output),
        ]
    else:
        device = 0 if arm == "tp1-d0" else 1
        command = _tp1_command(model=model, prompts=prompts, output=output, cell=cell)
    env = dict(os.environ)
    if arm in TP1_ARMS:
        env["HIP_VISIBLE_DEVICES"] = str(device)
    log = workdir / f"{arm}-{run_id}.log"
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=timeout_s
        )
    if completed.returncode != 0:
        raise MatchedBaselineError(
            f"{arm} failed rc={completed.returncode}; log={log} (see stderr there)"
        )
    payload = json.loads(output.read_text())
    if arm == "tp2":
        return {
            "arm": "tp2",
            "run_id": run_id,
            "command": shlex.join(command),
            "log": str(log),
            "device": None,
            "session_capture_ms": float(payload.get("session_capture_ms", 0.0)),
            "sweeps": payload["sweeps"],
            "devices": payload.get("devices", {}),
            "route": payload.get("route", {}),
        }
    return {
        "arm": arm,
        "run_id": run_id,
        "command": shlex.join(command),
        "log": str(log),
        "device": device,
        "prompt_metrics": payload["prompt_metrics"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--heldout-prompts", type=Path, default=DEFAULT_HELDOUT_PROMPTS)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--arm", choices=("tp2",), default=None, help="worker mode")
    parser.add_argument("--worker-reps", type=int, default=1)
    parser.add_argument("--arm-timeout", type=float, default=2400.0)
    parser.add_argument("--run", action="store_true", help="orchestrate the cell")
    args = parser.parse_args(argv)

    if args.arm == "tp2":
        payload = run_tp2_arm(
            model=args.model, prompts=args.prompts, cell=CELL_C1_NATURAL, reps=args.worker_reps
        )
        text = json.dumps(payload, indent=1) + "\n"
        if args.json:
            args.json.write_text(text)
        else:
            print(text)
        return 0

    if not args.run:
        parser.error("pass --run (orchestrator) or --arm tp2 (worker)")
    if args.reps < 3:
        parser.error("the declared cell requires >=3 independent reps")
    if args.workdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.workdir = Path(f"/tmp/tp2-matched-ar-{stamp}")
    args.workdir.mkdir(parents=True, exist_ok=True)

    gate_failures: list[str] = []
    accounting_failures: list[str] = []
    cell_rows, suite_failures = load_cell_prompts(args.prompts, args.heldout_prompts)
    gate_failures.extend(suite_failures)
    accounting_failures.extend(suite_failures)
    if not cell_rows:
        print("no cell prompts; aborting", flush=True)
        return 1
    cell_prompts = args.workdir / "cell_prompts.jsonl"
    write_prompt_file(cell_rows, cell_prompts)
    prompt_ids = [str(row["id"]) for row in cell_rows]
    print(f"cell {CELL_C1_NATURAL['name']}: {len(cell_rows)} prompts (natural length)", flush=True)

    arm_failures: dict[str, str] = {}
    arm_runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}

    def _attempt(arm: str, *, reps: int = 1) -> dict[str, Any] | None:
        try:
            run = run_arm(
                arm,
                model=args.model,
                prompts=cell_prompts,
                cell=CELL_C1_NATURAL,
                workdir=args.workdir,
                reps=reps,
                timeout_s=args.arm_timeout,
            )
        except (MatchedBaselineError, subprocess.SubprocessError) as error:
            arm_failures[arm] = f"{type(error).__name__}: {error}"
            print(f"arm {arm} FAILED: {arm_failures[arm]}", flush=True)
            return None
        arm_runs[arm].append(run)
        return run

    # Fault-safe order on this host: TP1 W7900, then TP2, then TP1 XTX last.
    declared_orders: dict[int, list[str]] = {}
    for rep in range(args.reps):
        order = ["tp1-d0", "tp2", "tp1-d1"] if rep % 2 == 0 else ["tp1-d0", "tp1-d1", "tp2"]
        declared_orders[rep] = order
        print(f"rep {rep}: order {order}", flush=True)
        for arm in order:
            _attempt(arm, reps=1)

    # Assemble independent reps. Each TP2 call above produced one sweep; TP1
    # arms produced one run each. A rep is complete only if every arm has an
    # execution that has not been reused in another rep.
    tp2_runs = arm_runs["tp2"]
    tp2_sweeps = [run["sweeps"][0] for run in tp2_runs if run.get("sweeps")]
    reps: list[dict[str, Any]] = []
    seen_run_ids: dict[str, str] = {}
    for rep in range(args.reps):
        run_ids: dict[str, str] = {}
        present: dict[str, Any] = {}
        for arm in ARMS:
            if arm == "tp2":
                if rep < len(tp2_sweeps):
                    present[arm] = {
                        "prompt_metrics": tp2_sweeps[rep],
                        "session_capture_ms": tp2_runs[rep].get("session_capture_ms", 0.0),
                    }
                    run_ids[arm] = tp2_runs[rep]["run_id"]
            else:
                runs = arm_runs[arm]
                if rep < len(runs):
                    present[arm] = {"prompt_metrics": runs[rep]["prompt_metrics"]}
                    run_ids[arm] = runs[rep]["run_id"]
        rep_entry: dict[str, Any] = {
            "rep": rep,
            "independent": True,
            "declared_order": declared_orders.get(rep, list(ARMS)),
            "run_ids": run_ids,
            "arms": {
                arm: {
                    **aggregate_rows(present[arm]["prompt_metrics"], arm=arm),
                    "categories": category_rows(present[arm]["prompt_metrics"], arm=arm),
                }
                for arm in present
            },
        }
        rep_entry["complete"] = all(arm in present for arm in ARMS)
        rep_failures = validate_rep(rep_entry, seen_run_ids=seen_run_ids)
        gate_failures.extend(rep_failures)
        accounting_failures.extend(rep_failures)
        for arm in present:
            for row in present[arm]["prompt_metrics"]:
                for failure in validate_accounting(arm, row, cell=CELL_C1_NATURAL):
                    gate_failures.append(f"rep{rep} {arm} {failure}")
                    accounting_failures.append(f"rep{rep} {arm} {failure}")
        reps.append(rep_entry)

    missing_arms = [arm for arm in ARMS if not arm_runs[arm]]
    if missing_arms:
        gate_failures.append(f"missing arms (blocked): {missing_arms}")
        accounting_failures.append(f"missing arms (blocked): {missing_arms}")

    # Qualification is separate from the timing ratio: none of these gates is
    # measured here, so the run can never be qualified, but a sound accounting
    # protocol still yields a diagnostic ratio (never a performance claim).
    qualification = {name: False for name in QUALIFICATION_GATES}
    qualification_failures = [
        f"qualification gate not measured: {name}"
        for name, measured in qualification.items()
        if not measured
    ]
    gate_failures.extend(qualification_failures)

    all_gates_passed = not gate_failures
    ratios = compute_ratios(reps) if not accounting_failures else None

    provenance: dict[str, Any] = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_revision": _git_revision(),
        "source_dirty": _git_dirty(),
        "host": _host_identity(),
        "model": {
            "path": str(args.model),
            "sha256": _model_sha256(args.model),
        },
        "arms": {
            arm: [
                {
                    "run_id": run["run_id"],
                    "command": run.get("command"),
                    "log": run.get("log"),
                    "device": run.get("device"),
                    "devices": run.get("devices"),
                    "route": run.get("route"),
                }
                for run in arm_runs[arm]
            ]
            for arm in ARMS
        },
    }
    artifact = {
        "schema": 1,
        "kind": "tp2_matched_ar_baseline",
        "status": "diagnostic" if not accounting_failures else "blocked",
        "cell": CELL_C1_NATURAL,
        "context_semantics": "natural_length (not a fixed context-128 cell)",
        "tp1_role": "matched-transfer diagnostic, not the optimized TP1 product baseline",
        "protocol": (
            "optimized TP1 resident graph replay vs TP2 graphed session, natural "
            "per-prompt context, greedy, no EOS stop, full logits per timed decode "
            "step on both arms, external whole-generation window with capture and "
            "destruction reported separately"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_suite": {
            "canonical": str(args.prompts),
            "heldout": str(args.heldout_prompts),
            "prompts": len(cell_rows),
            "prompt_ids": prompt_ids,
            "prompt_ids_sha256": _ids_hash([hash(pid) for pid in prompt_ids]),
        },
        "reps": reps,
        "ratios": ratios,
        "qualification": qualification,
        "qualified": all(qualification.values()) and all_gates_passed,
        "arm_failures": arm_failures,
        "missing_arms": missing_arms,
        "all_gates_passed": all_gates_passed,
        "accounting_failures": accounting_failures,
        "qualification_failures": qualification_failures,
        "gate_failures": gate_failures,
        "performance_claim": False,
        "provenance": provenance,
        "scope": (
            "diagnostic protocol only; not a qualified TP2 baseline and not a "
            "context-128 cell"
        ),
    }
    if args.json:
        args.json.write_text(json.dumps(artifact, indent=1) + "\n")
    ratio_text = (
        f"median={ratios['median_tp2_vs_faster_tp1']:.4f}" if ratios else "suppressed"
    )
    print(f"reps={len(reps)} ratio {ratio_text} gates={all_gates_passed}", flush=True)
    for failure in gate_failures[:20]:
        print(f"  FAIL {failure}", flush=True)
    return 0 if not accounting_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
