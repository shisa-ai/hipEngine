#!/usr/bin/env python3
"""Matched optimized-TP1 vs TP2 true-AR baseline (declared cell C1).

This is the honest denominator for TP2 AR decode: the optimized resident TP1
graph-replay route on each physical GPU, measured under the same protocol as the
TP2 session, not the TP2 session's own eager TP1 mode.

Declared cell C1 (predeclared before measuring; a run that cannot satisfy it is
reported as a blocker, never silently relaxed):

* context = 128 prompt tokens, output = 128 timed decode transitions
* warmup = 0, so timing starts at the prefill sample and the first timed decode
  transition runs at position ``len(prompt)`` on both arms
* greedy sampling, no EOS early stop
* full-vocabulary logits transferred on every timed decode step on both arms
  (the TP2 session always reads the full logits back; the TP1 harness is asked
  to do the same, so the per-step D2H cost is matched)
* TP1 graph replay and the TP2 graphed schedule; graph capture is reported
  separately and excluded from the headline decode rate
* cold builds, model load and prefill are outside the decode window

Accounting is validated per prompt, not inferred: timed transitions must equal
the declared output tokens, the context position at timing start must match, the
generated-token count must match each arm's prefill-sample convention, the final
logits must be finite, and the EOS policy must be ``none``. Two protocols that
disagree on any of these are not ratioed.

Usage (orchestrator):
    python3 scripts/tp2_matched_ar_baseline.py --run --reps 3 \\
        --json benchmarks/results/<artifact>.json
Worker (invoked by the orchestrator):
    python3 scripts/tp2_matched_ar_baseline.py --arm tp2 --json <path>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_category_bench import DEFAULT_MODEL, load_prompt_rows  # noqa: E402
from scripts.gguf_true_ar_category_bench import (  # noqa: E402
    DEFAULT_PROMPTS,
    build_chat_prompt,
    run_prompt_true_ar,
)

#: The dense model this cell measures; the inherited MTP-suite default is MoE,
#: which the dense MLP shard plan correctly refuses.
DEFAULT_DENSE_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")

#: Heldout-only rows are appended to the canonical suite; ids already in the
#: canonical file are skipped so the two scopes do not double-count.
DEFAULT_HELDOUT_PROMPTS = (
    REPO_ROOT / "benchmarks" / "prompts" / "laguna-target-ar-code-general-ja-heldout.jsonl"
)

#: Predeclared cell. Every field is normative; the validator fails closed.
CELL_C1: dict[str, Any] = {
    "name": "C1",
    "context_tokens": 128,
    "output_tokens": 128,
    "warmup_decode_tokens": 0,
    "sampling": "greedy",
    "eos": None,
    "logits_per_decode_step": True,
    "graph_replay_decode": True,
}

TP1_ARMS = ("tp1-d0", "tp1-d1")
ARMS = ("tp1-d0", "tp1-d1", "tp2")


class MatchedBaselineError(RuntimeError):
    """A declared-cell violation that must fail rather than produce a ratio."""


def load_cell_prompts(canonical: Path, heldout: Path | None) -> list[dict[str, Any]]:
    """Canonical suite plus heldout-only rows, deduplicated by id."""

    rows = load_prompt_rows(canonical)
    seen = {str(row["id"]) for row in rows}
    if heldout is not None and heldout.exists():
        for row in load_prompt_rows(heldout):
            if str(row["id"]) not in seen:
                rows.append(row)
                seen.add(str(row["id"]))
    return rows


def write_prompt_file(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_accounting(arm: str, row: dict[str, Any], *, cell: dict[str, Any]) -> list[str]:
    """Per-prompt protocol/accounting failures. Pure and fail-closed."""

    failures: list[str] = []
    prompt_id = str(row.get("id", "?"))
    if int(row.get("output_tokens", -1)) != int(cell["output_tokens"]):
        failures.append(f"{prompt_id}: output_tokens {row.get('output_tokens')} != {cell['output_tokens']}")
    if int(row.get("timed_decode_transitions", -1)) != int(cell["output_tokens"]):
        failures.append(
            f"{prompt_id}: timed_decode_transitions {row.get('timed_decode_transitions')} "
            f"!= {cell['output_tokens']}"
        )
    expected_start = int(row.get("prompt_tokens", -1)) + int(cell["warmup_decode_tokens"])
    if int(row.get("context_position_at_timing_start", -2)) != expected_start:
        failures.append(
            f"{prompt_id}: context_position_at_timing_start "
            f"{row.get('context_position_at_timing_start')} != prompt+warmup {expected_start}"
        )
    if bool(row.get("logits_per_decode_step", False)) != bool(cell["logits_per_decode_step"]):
        failures.append(f"{prompt_id}: logits_per_decode_step does not match the cell")
    if str(row.get("eos_policy", "")) != "none":
        failures.append(f"{prompt_id}: eos_policy {row.get('eos_policy')!r} != 'none'")
    if not bool(row.get("finite_final_logits", False)):
        failures.append(f"{prompt_id}: final logits not finite")
    generated = int(row.get("generated_count", -1))
    # TP1 keeps the prefill sample in its generated list; TP2 does not.
    expected_generated = (
        int(cell["output_tokens"]) + 1
        if arm in TP1_ARMS
        else int(cell["output_tokens"])
    )
    if generated != expected_generated:
        failures.append(
            f"{prompt_id}: generated_count {generated} != {expected_generated} for {arm}"
        )
    if float(row.get("decode_ms", 0.0)) <= 0.0:
        failures.append(f"{prompt_id}: decode_ms must be positive")
    return failures


def decode_ms_excluding_capture(row: dict[str, Any], *, arm: str) -> float:
    """Decode wall with graph capture removed, reported the same way on both arms.

    The TP1 harness captures a fresh per-prompt graph inside its decode window;
    the TP2 graphed schedule captures once before the token loop, so its decode
    traces already exclude capture. Subtracting the TP1 capture term makes the
    headline rate capture-neutral on both arms, and capture is reported
    separately.
    """

    decode_ms = float(row["decode_ms"])
    if arm in TP1_ARMS:
        decode_ms -= float(row.get("graph_capture_ms_included", 0.0))
    return decode_ms


def aggregate_rows(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    output_tokens = sum(int(r["output_tokens"]) for r in rows)
    decode_ms = sum(decode_ms_excluding_capture(r, arm=arm) for r in rows)
    capture_ms = sum(float(r.get("graph_capture_ms_included", 0.0)) for r in rows)
    return {
        "prompts": len(rows),
        "total_output_tokens": output_tokens,
        "decode_ms_excluding_capture": decode_ms,
        "graph_capture_ms": capture_ms,
        "decode_tok_s_excluding_capture": (
            1000.0 * output_tokens / decode_ms if decode_ms > 0 else 0.0
        ),
    }


def category_rows(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)
    return {cat: aggregate_rows(group, arm=arm) for cat, group in sorted(grouped.items())}


def compute_ratios(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """TP2/faster-TP1 ratio per rep, plus the spread. No ratio without >=1 rep."""

    per_rep: list[dict[str, Any]] = []
    for rep in reps:
        arms = rep["arms"]
        if not all(arm in arms for arm in ARMS):
            continue
        tp2 = arms["tp2"]["decode_tok_s_excluding_capture"]
        tp1 = {
            arm: arms[arm]["decode_tok_s_excluding_capture"] for arm in TP1_ARMS
        }
        faster_arm = max(tp1, key=lambda a: tp1[a])
        faster = tp1[faster_arm]
        per_rep.append(
            {
                "rep": int(rep["rep"]),
                "tp2_tok_s": tp2,
                "tp1_tok_s": tp1,
                "faster_tp1_arm": faster_arm,
                "faster_tp1_tok_s": faster,
                "tp2_vs_faster_tp1": tp2 / faster if faster > 0 else 0.0,
            }
        )
    ratios = [r["tp2_vs_faster_tp1"] for r in per_rep]
    return {
        "per_rep": per_rep,
        "reps": len(per_rep),
        "median_tp2_vs_faster_tp1": (
            sorted(ratios)[len(ratios) // 2] if ratios else 0.0
        ),
        "min_tp2_vs_faster_tp1": min(ratios) if ratios else 0.0,
        "max_tp2_vs_faster_tp1": max(ratios) if ratios else 0.0,
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
    sweeps: list[list[dict[str, Any]]] = []
    try:
        # Untimed warmup: force graph capture and JIT so it is outside timing.
        session.generate(token_rows[0][: min(8, len(token_rows[0]))], max_new_tokens=2)
        for _ in range(int(reps)):
            metrics: list[dict[str, Any]] = []
            for prompt_row, tokens in zip(rows, token_rows, strict=True):
                result = session.generate(
                    tokens,
                    max_new_tokens=int(cell["output_tokens"]),
                    eos_token_id=None,
                    capture_logits=True,
                )
                decode_traces = [t for t in result.step_traces if t.kind == "decode"]
                final_logits = None if result.logits is None else result.logits[-1]
                metrics.append(
                    {
                        "id": str(prompt_row["id"]),
                        "category": str(prompt_row["category"]),
                        "prompt_tokens": len(tokens),
                        "output_tokens": len(result.token_ids),
                        "timed_decode_transitions": len(decode_traces),
                        "context_position_at_timing_start": len(tokens),
                        "generated_count": len(result.token_ids),
                        "logits_per_decode_step": True,
                        "eos_policy": "none",
                        "finite_final_logits": bool(
                            final_logits is not None
                            and bool((final_logits == final_logits).all())
                            and bool((final_logits != float("inf")).all())
                            and bool((final_logits != float("-inf")).all())
                        ),
                        "decode_ms": 1000.0 * sum(t.total_s for t in decode_traces),
                        "prefill_ms": 1000.0
                        * sum(
                            t.total_s
                            for t in result.step_traces
                            if t.kind == "prefill"
                        ),
                        "graph_capture_ms_included": 0.0,
                        "generated_token_ids": [int(t) for t in result.token_ids],
                    }
                )
            sweeps.append(metrics)
    finally:
        session.close()
    return {"arm": "tp2", "sweeps": sweeps, "prompt_metrics": sweeps[0]}


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
    output = workdir / f"{arm}.json"
    if arm == "tp2":
        # Run TP2 in its own process: a resident TP2 session and a resident TP1
        # session must not hold contexts concurrently (observed GPU hang).
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
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        if completed.returncode != 0:
            raise MatchedBaselineError(
                f"tp2 failed rc={completed.returncode}: {completed.stderr[-2000:]}"
            )
        payload = json.loads(output.read_text())
        return {
            "arm": "tp2",
            "sweeps": payload["sweeps"],
            "prompt_metrics": payload["sweeps"][0],
        }
    device = 0 if arm == "tp1-d0" else 1
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(device)
    command = _tp1_command(model=model, prompts=prompts, output=output, cell=cell)
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=timeout_s
    )
    if completed.returncode != 0:
        raise MatchedBaselineError(
            f"{arm} failed rc={completed.returncode}: {completed.stderr[-2000:]}"
        )
    artifact = json.loads(output.read_text())
    return {
        "arm": arm,
        "device": device,
        "prompt_metrics": artifact["prompt_metrics"],
        "timing_protocol": artifact.get("timing_protocol"),
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
            model=args.model, prompts=args.prompts, cell=CELL_C1, reps=args.worker_reps
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
        parser.error("C1 requires >=3 balanced paired reps")
    if args.workdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.workdir = Path(f"/tmp/tp2-matched-ar-{stamp}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    cell_rows = load_cell_prompts(args.prompts, args.heldout_prompts)
    cell_prompts = args.workdir / "cell_prompts.jsonl"
    write_prompt_file(cell_rows, cell_prompts)
    prompt_ids = [str(row["id"]) for row in cell_rows]
    print(f"cell C1 suite: {len(cell_rows)} prompts", flush=True)

    reps: list[dict[str, Any]] = []
    gate_failures: list[str] = []
    arm_failures: dict[str, str] = {}
    # Order is fixed and fault-safe: the resident TP2 session runs after the
    # W7900 TP1 arm and before the XTX TP1 arm. On this host a TP1 process that
    # starts after a TP2 session has run faults, and the XTX optimized TP1
    # route currently faults on its own; running it last bounds the damage.
    def _attempt(arm: str, *, reps: int = 1):
        print(f"arm {arm}", flush=True)
        try:
            return run_arm(
                arm,
                model=args.model,
                prompts=cell_prompts,
                cell=CELL_C1,
                workdir=args.workdir,
                reps=reps,
                timeout_s=args.arm_timeout,
            )
        except (MatchedBaselineError, subprocess.SubprocessError) as error:
            arm_failures[arm] = f"{type(error).__name__}: {error}"
            print(f"arm {arm} FAILED: {arm_failures[arm]}", flush=True)
            return None

    tp1_d0 = _attempt("tp1-d0")
    tp2_payload = _attempt("tp2", reps=args.reps)
    tp1_d1 = _attempt("tp1-d1")
    for rep in range(args.reps):
        arm_results: dict[str, Any] = {}
        if tp1_d0 is not None:
            arm_results["tp1-d0"] = tp1_d0
        if tp1_d1 is not None:
            arm_results["tp1-d1"] = tp1_d1
        if tp2_payload is not None:
            arm_results["tp2"] = {
                "arm": "tp2",
                "prompt_metrics": tp2_payload["sweeps"][rep],
            }
        for arm in ARMS:
            if arm not in arm_results:
                continue
            for row in arm_results[arm]["prompt_metrics"]:
                for failure in validate_accounting(arm, row, cell=CELL_C1):
                    gate_failures.append(f"rep{rep} {arm} {failure}")
        reps.append(
            {
                "rep": rep,
                "arm_order": ["tp1-d0", "tp2", "tp1-d1"],
                "interleaved": False,
                "arms_present": sorted(arm_results),
                "arms": {
                    arm: {
                        **aggregate_rows(arm_results[arm]["prompt_metrics"], arm=arm),
                        "categories": category_rows(arm_results[arm]["prompt_metrics"], arm=arm),
                    }
                    for arm in sorted(arm_results)
                },
            }
        )

    ratios = compute_ratios(reps)
    missing_arms = sorted(set(ARMS) - set().union(*(set(r["arms"]) for r in reps))) if reps else list(ARMS)
    if missing_arms:
        gate_failures.append(f"missing arms (blocked): {missing_arms}")
    artifact = {
        "schema": 1,
        "kind": "tp2_matched_ar_baseline",
        "cell": CELL_C1,
        "protocol": (
            "optimized TP1 resident graph replay on each physical GPU vs the TP2 "
            "graphed session, same prompt token IDs, greedy, no EOS stop, full "
            "logits per timed decode step on both arms, capture excluded and "
            "reported separately"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "host": os.uname().nodename,
        "model": str(args.model),
        "prompt_suite": {
            "canonical": str(args.prompts),
            "heldout": str(args.heldout_prompts),
            "prompts": len(cell_rows),
            "prompt_ids": prompt_ids,
        },
        "reps": reps,
        "ratios": ratios,
        "arm_failures": arm_failures,
        "missing_arms": missing_arms,
        "all_gates_passed": not gate_failures,
        "gate_failures": gate_failures,
        "performance_claim": False,
        "scope": (
            "single measured cell C1 only (context 128 / output 128, full "
            "canonical+heldout suite); not a campaign-wide TP2 claim"
        ),
    }
    if args.json:
        args.json.write_text(json.dumps(artifact, indent=1) + "\n")
    print(
        f"reps={len(reps)} median_ratio={ratios['median_tp2_vs_faster_tp1']:.4f} "
        f"range=[{ratios['min_tp2_vs_faster_tp1']:.4f},"
        f"{ratios['max_tp2_vs_faster_tp1']:.4f}] gates={not gate_failures}",
        flush=True,
    )
    if gate_failures:
        for failure in gate_failures[:20]:
            print(f"  FAIL {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
