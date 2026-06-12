#!/usr/bin/env python3
"""Measure MTP verifier economics in AR-token-equivalent units.

This is the M12 entry-point diagnostic.  llama.cpp-style MTP speedups come from
keeping one speculative target-verify cycle near ~2 AR-token equivalents while
emitting ~2-4 visible tokens.  Kernel family milliseconds are useful for local
work, but the go/no-go metric for MTP is:

    cycle_cost_ar_tokens = avg_mtp_verify_cycle_wall_ms / ar_decode_ms_per_token

A candidate budget B can only beat AR when:

    avg_visible_tokens_per_verify_cycle > cycle_cost_ar_tokens

and a 1.5x row needs:

    avg_visible_tokens_per_verify_cycle / cycle_cost_ar_tokens >= 1.5

This wrapper runs scripts/mtp_chain_e2e_smoke.py for one or more candidate
budgets and computes those ratios from the smoke JSON.  It intentionally keeps
``performance_claim=false``: the output is an economics diagnostic used to pick
the next verifier-loop architecture change, not a retained speed row.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_PROMPT_TOKENS = "151646"
DEFAULT_BUDGETS = (1, 2, 3, 5)


def _split_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return statistics.fmean(vals) if vals else None


def _std(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0 if vals else None
    return statistics.stdev(vals)


def _safe_div(numer: float, denom: float) -> float | None:
    if denom == 0:
        return None
    return float(numer) / float(denom)


def _cycle_wall_seconds(mtp: dict[str, Any]) -> tuple[float, list[float]]:
    markers = mtp.get("cycle_marker_ns") or []
    per_cycle: list[float] = []
    for marker in markers:
        start = int(marker.get("start_perf_ns", 0))
        end = int(marker.get("end_perf_ns", 0))
        if end > start:
            per_cycle.append((end - start) / 1e9)
    if per_cycle:
        return float(sum(per_cycle)), per_cycle
    # Older smoke JSONs may not have per-cycle markers.  Fall back to the whole
    # decode window; this includes terminal AR cleanup and is therefore a
    # conservative overestimate of speculative cycle cost.
    decode_seconds = float(mtp.get("decode_seconds") or 0.0)
    accepted = mtp.get("accepted_lengths") or []
    cycles = max(1, len(accepted))
    return decode_seconds, [decode_seconds / cycles for _ in range(cycles)]


def _economics_from_smoke(smoke: dict[str, Any], *, llama_target_cycle_cost: float) -> dict[str, Any]:
    ar = smoke.get("ar") or {}
    mtp = smoke.get("mtp") or {}
    decode_tokens = int(smoke.get("decode_tokens") or 0)
    budget = int(smoke.get("candidate_budget") or 0)
    accepted = [int(x) for x in (mtp.get("accepted_lengths") or [])]
    active_budgets = [int(x) for x in (mtp.get("active_budgets") or [])]
    cycles = len(accepted)
    if cycles <= 0:
        raise ValueError("smoke JSON contains no MTP verifier cycles")

    ar_decode_seconds = float(ar.get("decode_seconds") or 0.0)
    mtp_decode_seconds = float(mtp.get("decode_seconds") or 0.0)
    ar_token_seconds = ar_decode_seconds / decode_tokens if decode_tokens > 0 else 0.0
    cycle_wall_total, cycle_wall_per_cycle = _cycle_wall_seconds(mtp)
    verify_seconds = float(mtp.get("verify_seconds") or 0.0)
    proposal_update_seconds = float(mtp.get("proposal_decode_update_seconds") or 0.0)
    proposal_prefill_seconds = float(mtp.get("proposal_prefill_seconds") or 0.0)
    proposal_snapshot_saves = int(mtp.get("proposal_snapshot_saves") or 0)
    proposal_snapshot_skips = int(mtp.get("proposal_snapshot_skips") or 0)
    ar_fallback_cycles = int(mtp.get("ar_fallback_cycles") or 0)
    ar_fallback_tokens = int(mtp.get("ar_fallback_tokens") or 0)
    ar_fallback_seconds = float(mtp.get("ar_fallback_seconds") or 0.0)
    ar_fallback_proposer_update_seconds = float(mtp.get("ar_fallback_proposer_update_seconds") or 0.0)
    confidence_ar_fallback_cycles = int(mtp.get("confidence_ar_fallback_cycles") or 0)
    confidence_ar_fallback_tokens = int(mtp.get("confidence_ar_fallback_tokens") or 0)

    # Each successful speculative verify cycle commits the root target token plus
    # the accepted draft prefix.  The harness may finish with a terminal AR token
    # when remaining == 1; keep that separate from the speculative-cycle average.
    visible_tokens_from_verify_cycles = sum(1 + x for x in accepted)
    terminal_or_clipped_tokens = int(decode_tokens) - int(visible_tokens_from_verify_cycles)
    total_active_budget = sum(active_budgets) if active_budgets else cycles * budget
    avg_active_budget = total_active_budget / cycles if cycles else 0.0
    avg_visible_tokens = visible_tokens_from_verify_cycles / cycles
    avg_accepted = sum(accepted) / cycles
    acceptance_rate = (sum(accepted) / total_active_budget) if total_active_budget else 0.0

    avg_cycle_wall_seconds = cycle_wall_total / cycles
    avg_verify_seconds = verify_seconds / cycles
    avg_proposal_update_seconds = proposal_update_seconds / cycles
    ar_decode_ms_per_token = ar_token_seconds * 1000.0
    cycle_cost_ar_tokens = (avg_cycle_wall_seconds / ar_token_seconds) if ar_token_seconds > 0 else None
    verify_cost_ar_tokens = (avg_verify_seconds / ar_token_seconds) if ar_token_seconds > 0 else None
    proposal_update_cost_ar_tokens = (avg_proposal_update_seconds / ar_token_seconds) if ar_token_seconds > 0 else None

    observed_cycle_speedup = _safe_div(avg_visible_tokens, cycle_cost_ar_tokens or 0.0) if cycle_cost_ar_tokens else None
    actual_decode_speedup = _safe_div(float(mtp.get("decode_tok_s") or 0.0), float(ar.get("decode_tok_s") or 0.0))
    perfect_accept_ceiling = _safe_div(budget + 1, cycle_cost_ar_tokens or 0.0) if cycle_cost_ar_tokens else None
    required_avg_visible_for_break_even = cycle_cost_ar_tokens
    required_avg_accept_for_break_even = max(0.0, (cycle_cost_ar_tokens or 0.0) - 1.0) if cycle_cost_ar_tokens else None
    required_acceptance_for_break_even = (
        _safe_div(required_avg_accept_for_break_even or 0.0, avg_active_budget) if avg_active_budget else None
    )
    target_cycle_cost_for_observed_accept_1x = avg_visible_tokens
    target_cycle_cost_for_observed_accept_1p5x = avg_visible_tokens / 1.5

    return {
        "exact_ar_match": bool(smoke.get("exact_ar_match")),
        "status": smoke.get("status"),
        "decode_tokens": decode_tokens,
        "candidate_budget": budget,
        "cycles": cycles,
        "active_budgets": active_budgets,
        "accepted_lengths": accepted,
        "avg_active_budget": avg_active_budget,
        "avg_accepted_per_cycle": avg_accepted,
        "avg_visible_tokens_per_cycle": avg_visible_tokens,
        "acceptance_rate": acceptance_rate,
        "visible_tokens_from_verify_cycles": visible_tokens_from_verify_cycles,
        "terminal_or_clipped_tokens": terminal_or_clipped_tokens,
        "ar_decode_tok_s": ar.get("decode_tok_s"),
        "mtp_decode_tok_s": mtp.get("decode_tok_s"),
        "actual_decode_speedup_vs_ar": actual_decode_speedup,
        "ar_decode_ms_per_token": ar_decode_ms_per_token,
        "mtp_decode_ms_per_token": (1000.0 / float(mtp.get("decode_tok_s"))) if mtp.get("decode_tok_s") else None,
        "cycle_wall_ms_per_cycle": avg_cycle_wall_seconds * 1000.0,
        "cycle_wall_ms_per_cycle_values": [v * 1000.0 for v in cycle_wall_per_cycle],
        "verify_ms_per_cycle": avg_verify_seconds * 1000.0,
        "proposal_update_ms_per_cycle": avg_proposal_update_seconds * 1000.0,
        "proposal_prefill_seconds": proposal_prefill_seconds,
        "proposal_snapshot_saves": proposal_snapshot_saves,
        "proposal_snapshot_skips": proposal_snapshot_skips,
        "proposal_snapshot_saves_per_cycle": proposal_snapshot_saves / cycles,
        "proposal_snapshot_skips_per_cycle": proposal_snapshot_skips / cycles,
        "ar_fallback_zero_streak": int(mtp.get("ar_fallback_zero_streak") or 0),
        "ar_fallback_tokens_per_window": int(mtp.get("ar_fallback_tokens_per_window") or 1),
        "ar_fallback_until_end": bool(mtp.get("ar_fallback_until_end")),
        "ar_fallback_cycles": ar_fallback_cycles,
        "ar_fallback_tokens": ar_fallback_tokens,
        "ar_fallback_seconds": ar_fallback_seconds,
        "ar_fallback_proposer_update_seconds": ar_fallback_proposer_update_seconds,
        "ar_fallback_ms_per_cycle": (ar_fallback_seconds / ar_fallback_cycles * 1000.0) if ar_fallback_cycles else 0.0,
        "ar_fallback_proposer_update_ms_per_cycle": (
            ar_fallback_proposer_update_seconds / ar_fallback_cycles * 1000.0
        ) if ar_fallback_cycles else 0.0,
        "confidence_threshold": float(mtp.get("confidence_threshold") or 0.0),
        "confidence_ar_fallback_cycles": confidence_ar_fallback_cycles,
        "confidence_ar_fallback_tokens": confidence_ar_fallback_tokens,
        "cycle_cost_ar_tokens": cycle_cost_ar_tokens,
        "verify_cost_ar_tokens": verify_cost_ar_tokens,
        "proposal_update_cost_ar_tokens": proposal_update_cost_ar_tokens,
        "observed_cycle_speedup_vs_ar": observed_cycle_speedup,
        "perfect_accept_speedup_ceiling_vs_ar": perfect_accept_ceiling,
        "required_avg_visible_tokens_for_1x": required_avg_visible_for_break_even,
        "required_avg_accepted_for_1x": required_avg_accept_for_break_even,
        "required_acceptance_rate_for_1x": required_acceptance_for_break_even,
        "target_cycle_cost_ar_tokens_for_observed_accept_1x": target_cycle_cost_for_observed_accept_1x,
        "target_cycle_cost_ar_tokens_for_observed_accept_1p5x": target_cycle_cost_for_observed_accept_1p5x,
        "llama_like_cycle_cost_target_ar_tokens": llama_target_cycle_cost,
        "llama_like_gap_multiplier": _safe_div(cycle_cost_ar_tokens or 0.0, llama_target_cycle_cost),
        "proposal_trace_sample": mtp.get("proposal_trace_sample"),
        "acceptance_diagnostics": mtp.get("acceptance_diagnostics"),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        "avg_active_budget",
        "avg_accepted_per_cycle",
        "avg_visible_tokens_per_cycle",
        "acceptance_rate",
        "ar_decode_tok_s",
        "mtp_decode_tok_s",
        "actual_decode_speedup_vs_ar",
        "ar_decode_ms_per_token",
        "cycle_wall_ms_per_cycle",
        "verify_ms_per_cycle",
        "proposal_update_ms_per_cycle",
        "cycle_cost_ar_tokens",
        "verify_cost_ar_tokens",
        "proposal_update_cost_ar_tokens",
        "proposal_snapshot_saves",
        "proposal_snapshot_skips",
        "proposal_snapshot_saves_per_cycle",
        "proposal_snapshot_skips_per_cycle",
        "ar_fallback_cycles",
        "ar_fallback_tokens",
        "ar_fallback_seconds",
        "ar_fallback_proposer_update_seconds",
        "ar_fallback_ms_per_cycle",
        "ar_fallback_proposer_update_ms_per_cycle",
        "confidence_threshold",
        "confidence_ar_fallback_cycles",
        "confidence_ar_fallback_tokens",
        "observed_cycle_speedup_vs_ar",
        "perfect_accept_speedup_ceiling_vs_ar",
        "required_avg_visible_tokens_for_1x",
        "required_avg_accepted_for_1x",
        "required_acceptance_rate_for_1x",
        "target_cycle_cost_ar_tokens_for_observed_accept_1x",
        "target_cycle_cost_ar_tokens_for_observed_accept_1p5x",
        "llama_like_gap_multiplier",
    ]
    aggregate: dict[str, Any] = {
        "runs": len(runs),
        "all_exact_ar_match": all(bool(r.get("exact_ar_match")) for r in runs),
        "accepted_lengths_by_run": [r.get("accepted_lengths") for r in runs],
        "active_budgets_by_run": [r.get("active_budgets") for r in runs],
        "acceptance_diagnostics_by_run": [r.get("acceptance_diagnostics") for r in runs if r.get("acceptance_diagnostics")],
    }
    for field in numeric_fields:
        vals = [r[field] for r in runs if r.get(field) is not None]
        aggregate[f"{field}_mean"] = _mean(vals)
        aggregate[f"{field}_std"] = _std(vals)
    return aggregate


def _run_one(
    args: argparse.Namespace,
    *,
    budget: int,
    run_idx: int,
    prompt_tokens: str,
    raw_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    json_path = raw_root / f"mtp-economics-B{budget}-run{run_idx}.json"
    log_path = raw_root / f"mtp-economics-B{budget}-run{run_idx}.log"
    cmd = [
        sys.executable,
        "scripts/mtp_chain_e2e_smoke.py",
        "--model",
        str(args.model),
        "--prompt-tokens",
        prompt_tokens,
        "--decode-tokens",
        str(args.decode_tokens),
        "--candidate-budget",
        str(budget),
        "--proposal-impl",
        str(args.proposal_impl),
        "--backend",
        str(args.backend),
        "--chain-attn-mode",
        str(args.chain_attn_mode),
        "--graph-mode",
        str(args.graph_mode),
        "--json",
        str(json_path),
    ]
    if int(getattr(args, "active_budget_cap", 0)) > 0:
        cmd += ["--active-budget-cap", str(int(args.active_budget_cap))]
    if bool(getattr(args, "acceptance_diagnostics", False)):
        cmd.append("--acceptance-diagnostics")
    if float(getattr(args, "confidence_threshold", 0.0) or 0.0) > 0.0:
        cmd += ["--confidence-threshold", str(float(args.confidence_threshold))]
    if int(getattr(args, "ar_fallback_zero_streak", 0)) > 0:
        cmd += [
            "--ar-fallback-zero-streak",
            str(int(args.ar_fallback_zero_streak)),
            "--ar-fallback-tokens",
            str(int(args.ar_fallback_tokens)),
        ]
        if bool(getattr(args, "ar_fallback_until_end", False)):
            cmd.append("--ar-fallback-until-end")
    env = os.environ.copy()
    if args.small_batch_decode_threshold is not None:
        env["HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD"] = str(args.small_batch_decode_threshold)
    if args.verify_gpu_accept is not None:
        env["HIPENGINE_VERIFY_GPU_ACCEPT"] = str(args.verify_gpu_accept)
    if args.hip_arch:
        env["HIPENGINE_HIP_ARCH"] = str(args.hip_arch)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        raise RuntimeError(
            f"smoke failed for B={budget} run={run_idx} with exit {completed.returncode}; "
            f"tail of {log_path}:\n" + "\n".join(tail)
        )
    smoke = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = _economics_from_smoke(smoke, llama_target_cycle_cost=float(args.llama_target_cycle_cost))
    metrics["run_idx"] = int(run_idx)
    metrics["smoke_json"] = str(json_path)
    metrics["smoke_log"] = str(log_path)
    metrics["subprocess_wall_seconds"] = wall_seconds
    metrics["command"] = " ".join(cmd)
    return smoke, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--prompt-tokens-file", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--candidate-budgets", default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument(
        "--active-budget-cap",
        type=int,
        default=0,
        help=(
            "Forward --active-budget-cap to mtp_chain_e2e_smoke.py. Diagnostic "
            "only: caps active drafted candidates while keeping verifier rows "
            "at the selected --candidate-budgets value."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--proposal-impl", choices=("persistent_device", "persistent_device_b1", "reload_d2h"), default="persistent_device")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="c1_loop")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--small-batch-decode-threshold", type=int, default=7)
    parser.add_argument("--verify-gpu-accept", default=None)
    parser.add_argument(
        "--acceptance-diagnostics",
        action="store_true",
        help="Forward --acceptance-diagnostics to mtp_chain_e2e_smoke.py and retain the diagnostics in this artifact.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=(
            "Forward opt-in MTP whole-cycle confidence gate: for persistent "
            "chain mode, route a cycle through exact AR when the depth-1 MTP "
            "top-1 probability proxy is below this threshold. 0 disables."
        ),
    )
    parser.add_argument(
        "--ar-fallback-zero-streak",
        type=int,
        default=0,
        help=(
            "Forward opt-in B=1/AR fallback policy: after this many consecutive "
            "zero-accept MTP cycles, skip the next --ar-fallback-tokens through "
            "target AR. 0 disables."
        ),
    )
    parser.add_argument(
        "--ar-fallback-tokens",
        type=int,
        default=1,
        help="Number of target AR tokens to emit per --ar-fallback-zero-streak trigger.",
    )
    parser.add_argument(
        "--ar-fallback-until-end",
        action="store_true",
        help=(
            "When --ar-fallback-zero-streak triggers, finish remaining decode with "
            "plain target AR instead of resuming MTP."
        ),
    )
    parser.add_argument("--llama-target-cycle-cost", type=float, default=2.0)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-mtp-verifier-economics"))
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / f"{date.today().isoformat()}-hipengine-mtp-verifier-economics-m12.json",
    )
    args = parser.parse_args()

    if args.prompt_tokens_file is not None:
        prompt_tokens = args.prompt_tokens_file.read_text(encoding="utf-8").strip()
    else:
        prompt_tokens = str(args.prompt_tokens).strip()
    budgets = _split_csv_ints(args.candidate_budgets)
    if not budgets:
        raise SystemExit("--candidate-budgets must contain at least one integer")
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    by_budget: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    for budget in budgets:
        runs: list[dict[str, Any]] = []
        for run_idx in range(1, int(args.runs) + 1):
            print(f"[economics] B={budget} run={run_idx}/{args.runs}", flush=True)
            _smoke, metrics = _run_one(args, budget=budget, run_idx=run_idx, prompt_tokens=prompt_tokens, raw_root=args.raw_root)
            runs.append(metrics)
            all_runs.append(metrics)
            print(
                "  "
                f"exact={metrics['exact_ar_match']} "
                f"mtp/ar={metrics['actual_decode_speedup_vs_ar']:.3f} "
                f"cycle_cost={metrics['cycle_cost_ar_tokens']:.2f} AR-tok "
                f"emit/cycle={metrics['avg_visible_tokens_per_cycle']:.2f} "
                f"perfect_ceiling={metrics['perfect_accept_speedup_ceiling_vs_ar']:.2f}x",
                flush=True,
            )
        by_budget[str(budget)] = {
            "budget": budget,
            "runs": runs,
            "aggregate": _aggregate_runs(runs),
        }

    artifact = {
        "schema": 1,
        "status": "diagnostic_retained",
        "performance_claim": False,
        "phase": "M12.0",
        "date": date.today().isoformat(),
        "purpose": "M12 verifier-loop economics: measure speculative cycle cost in AR-token-equivalent units by candidate budget.",
        "model": str(args.model),
        "backend": str(args.backend),
        "hip_arch": str(args.hip_arch),
        "prompt_tokens": prompt_tokens,
        "decode_tokens": int(args.decode_tokens),
        "candidate_budgets": budgets,
        "active_budget_cap": int(args.active_budget_cap),
        "runs_per_budget": int(args.runs),
        "proposal_impl": str(args.proposal_impl),
        "chain_attn_mode": str(args.chain_attn_mode),
        "graph_mode": str(args.graph_mode),
        "confidence_threshold": float(args.confidence_threshold),
        "small_batch_decode_threshold": int(args.small_batch_decode_threshold) if args.small_batch_decode_threshold is not None else None,
        "verify_gpu_accept": args.verify_gpu_accept,
        "acceptance_diagnostics": bool(args.acceptance_diagnostics),
        "llama_target_cycle_cost_ar_tokens": float(args.llama_target_cycle_cost),
        "go_no_go_rule": {
            "beats_ar": "avg_visible_tokens_per_verify_cycle > cycle_cost_ar_tokens",
            "hits_1p5x": "avg_visible_tokens_per_verify_cycle / cycle_cost_ar_tokens >= 1.5",
            "llama_like_ratio": "cycle_cost_ar_tokens <= ~2.0 for B=2/3-class small-depth MTP",
        },
        "by_budget": by_budget,
    }
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[economics] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
