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


def _sha256_json(value: Any) -> str:
    """Deterministic content hash: UTF-8 JSON with sorted keys.

    Never uses Python's ``hash()``, which is randomized per process by
    ``PYTHONHASHSEED`` and would make artifacts incomparable across runs.
    """

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prompt_ids_sha256(prompt_ids: Sequence[Any]) -> str:
    return _sha256_json([str(pid) for pid in prompt_ids])


def token_tuple_sha256(tokens: Sequence[Any]) -> str:
    return _sha256_json([int(t) for t in tokens])


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
        return result if result == value else None
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_accounting(arm: str, row: dict[str, Any], *, cell: dict[str, Any]) -> list[str]:
    """Per-prompt protocol/accounting failures. Pure and fail-closed.

    Every numeric read goes through a safe converter so a missing/``None``/NaN
    field produces a failure string rather than raising during aggregation.
    """

    failures: list[str] = []
    prompt_id = str(row.get("id", "?"))
    output = int(cell["output_tokens"])

    def need(condition: bool, message: str) -> None:
        if not condition:
            failures.append(f"{prompt_id}: {message}")

    prompt_tokens = _as_int(row.get("prompt_tokens"))
    context_tokens = _as_int(row.get("context_tokens"))
    # -- declared work ------------------------------------------------------
    need(_as_int(row.get("output_tokens")) == output, f"output_tokens != {output}")
    need(
        _as_int(row.get("timed_decode_transitions")) == output,
        f"timed_decode_transitions != {output}",
    )
    # -- context semantics --------------------------------------------------
    context = cell["context"]
    if context == "natural":
        need(
            context_tokens is not None and context_tokens == prompt_tokens,
            "natural-length context_tokens != prompt_tokens",
        )
    else:
        need(prompt_tokens == int(context), f"prompt_tokens != declared context {context}")
        need(False, "fixed-length cell construction is not implemented")
    need(prompt_tokens is not None and prompt_tokens > 0, "prompt_tokens must be positive")
    position = _as_int(row.get("context_position_at_timing_start"))
    need(
        position is not None
        and prompt_tokens is not None
        and position == prompt_tokens + int(cell["warmup_decode_tokens"]),
        "context_position_at_timing_start != prompt_tokens + warmup",
    )
    # -- route --------------------------------------------------------------
    need(row.get("graph_effective") is True, "graph replay not effective")
    need(
        row.get("logits_per_decode_step") is cell["logits_per_decode_step"],
        "logits_per_decode_step does not match the cell",
    )
    # -- logits -------------------------------------------------------------
    need(row.get("finite_final_logits") is True, "final logits not finite")
    need(
        row.get("finite_all_decode_logits") is True,
        "all decode-step logits not verified finite",
    )
    need(str(row.get("eos_policy", "")) == "none", "eos_policy != 'none'")
    # -- sampled-output alignment (not counts alone) ------------------------
    sampled = row.get("sampled_output_ids")
    if not isinstance(sampled, list):
        failures.append(f"{prompt_id}: sampled_output_ids missing")
    else:
        need(len(sampled) == output, f"sampled_output_ids {len(sampled)} != {output}")
        try:
            need(row.get("sampled_output_sha256") == _ids_hash(sampled), "sampled_output_sha256 mismatch")
        except (TypeError, ValueError):
            failures.append(f"{prompt_id}: sampled_output_ids not integer-valued")
    need(row.get("prefill_sample_id") is not None, "prefill_sample_id missing")
    need(
        isinstance(row.get("prompt_token_sha256"), str) and bool(row.get("prompt_token_sha256")),
        "prompt_token_sha256 missing",
    )
    # -- timing windows -----------------------------------------------------
    for key in ("total_generation_ms", "capture_ms", "destroy_ms"):
        value = _as_float(row.get(key))
        if value is None or value < 0:
            failures.append(f"{prompt_id}: {key} must be finite and non-negative")
    total = _as_float(row.get("total_generation_ms"))
    capture = _as_float(row.get("capture_ms"))
    destroy = _as_float(row.get("destroy_ms"))
    if total is not None and capture is not None and destroy is not None:
        adjusted = total - capture - destroy
        need(adjusted > 0.0, "adjusted window must be positive")
    prefill = _as_float(row.get("prefill_ms"))
    need(prefill is not None and prefill >= 0.0, "prefill_ms must be finite and non-negative")
    return failures


def validate_arm_rows(
    arm: str,
    rows: list[dict[str, Any]],
    *,
    expected_ids: Sequence[str],
    expected_categories: dict[str, str],
) -> list[str]:
    """Returned prompt coverage/order/category vs the declared suite.

    Rejects zero rows, duplicate/missing/extra ids, reordered ids, and changed
    category membership. Cross-arm token-hash equality is a separate check.
    """

    failures: list[str] = []
    if not rows:
        return [f"{arm}: returned zero prompt rows"]
    ids = [str(r.get("id", "")) for r in rows]
    if len(set(ids)) != len(ids):
        failures.append(f"{arm}: duplicate prompt ids in returned rows")
    if ids != list(expected_ids):
        missing = [i for i in expected_ids if i not in set(ids)]
        extra = [i for i in ids if i not in set(expected_ids)]
        failures.append(
            f"{arm}: prompt id coverage/order mismatch "
            f"(returned={len(ids)} expected={len(expected_ids)} missing={missing[:4]} extra={extra[:4]})"
        )
    for row in rows:
        pid = str(row.get("id", ""))
        if pid not in expected_categories:
            continue
        if str(row.get("category")) != expected_categories[pid]:
            failures.append(
                f"{arm}: {pid} category {row.get('category')!r} != {expected_categories[pid]!r}"
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
    total = _as_float(row.get("total_generation_ms"))
    if total is None:
        raise MatchedBaselineError("adjusted_ms requires a finite total_generation_ms")
    capture = _as_float(row.get("capture_ms")) or 0.0
    destroy = _as_float(row.get("destroy_ms")) or 0.0
    return total - capture - destroy


def cross_arm_token_hashes(
    per_arm_rows: dict[str, list[dict[str, Any]]], *, expected_ids: Sequence[str]
) -> list[str]:
    """Require every arm to have tokenized each prompt identically.

    This is a tokenizer/input-identity check only. Cross-arm *generated* token
    equality is deliberately NOT required: TP1 and TP2 drift numerically under
    the production numerical contract.
    """

    failures: list[str] = []
    by_arm: dict[str, dict[str, str]] = {}
    for arm, rows in per_arm_rows.items():
        by_arm[arm] = {str(r.get("id", "")): r.get("prompt_token_sha256") for r in rows}
    for pid in expected_ids:
        hashes = {arm: table.get(pid) for arm, table in by_arm.items()}
        values = {h for h in hashes.values() if h}
        if len(values) != 1:
            failures.append(
                f"prompt {pid}: prompt_token_sha256 differs across arms: {hashes}"
            )
    return failures


def aggregate_rows(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    output_tokens = sum(_as_int(r.get("output_tokens")) or 0 for r in rows)
    adjusted = 0.0
    total = 0.0
    capture = 0.0
    destroy = 0.0
    for row in rows:
        adjusted += adjusted_ms(row)
        total += _as_float(row.get("total_generation_ms")) or 0.0
        capture += _as_float(row.get("capture_ms")) or 0.0
        destroy += _as_float(row.get("destroy_ms")) or 0.0
    return {
        "prompts": len(rows),
        "total_output_tokens": output_tokens,
        "adjusted_ms": adjusted,
        "total_generation_ms": total,
        "capture_ms": capture,
        "destroy_ms": destroy,
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
        if not all(math.isfinite(rate) and rate > 0 for rate in [tp2, *tp1.values()]):
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
    session_destroy_ms = 0.0
    devices: dict[str, Any] = {}
    route: dict[str, Any] = {}
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
                        "prompt_token_sha256": token_tuple_sha256(tokens),
                        "total_generation_ms": total_ms,
                        "prefill_ms": 1000.0
                        * sum(t.total_s for t in result.step_traces if t.kind == "prefill"),
                        "capture_ms": 0.0,
                        "destroy_ms": 0.0,
                    }
                )
            sweeps.append(metrics)
        devices = _device_identities(session)
        route = _resolved_route(session)
    finally:
        destroy_start = time.perf_counter()
        session.close()
        session_destroy_ms = 1000.0 * (time.perf_counter() - destroy_start)
    return {
        "arm": "tp2",
        "session_capture_ms": session_capture_ms,
        "session_destroy_ms": session_destroy_ms,
        "sweeps": sweeps,
        "prompt_metrics": sweeps[0] if sweeps else [],
        "devices": devices,
        "route": route,
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
            "session_destroy_ms": float(payload.get("session_destroy_ms", 0.0)),
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
        # TP1 physical identities, execution profile, KV policy and variant
        # manifest live in the child's own artifact provenance; persist it here
        # so the orchestrator artifact is self-contained.
        "tp1_provenance": payload.get("provenance"),
        "tp1_timing_protocol": payload.get("timing_protocol"),
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
    parser.add_argument('--protocol', choices=('matched-transfer', 'product'), default='matched-transfer')
    parser.add_argument('--product-arm', choices=ARMS)
    parser.add_argument('--rep-index', type=int, default=0)
    parser.add_argument('--capacity-only', action='store_true')
    parser.add_argument('--quality-json', type=Path)
    args = parser.parse_args(argv)
    if args.protocol == 'product':
        if not args.json:
            parser.error('product protocol requires --json')
        if args.product_arm:
            return product_worker(args)
        if not args.run or args.reps < 3:
            parser.error('product campaign requires --run and >=3 independent repetitions')
        return run_product_campaign(args)

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
    matching_failures: list[str] = []
    provenance_gaps: list[str] = []
    cell_rows, suite_failures = load_cell_prompts(args.prompts, args.heldout_prompts)
    gate_failures.extend(suite_failures)
    accounting_failures.extend(suite_failures)
    suite_ok = bool(cell_rows)
    expected_ids = [str(row["id"]) for row in cell_rows]
    expected_categories = {str(row["id"]): str(row["category"]) for row in cell_rows}
    cell_prompts = args.workdir / "cell_prompts.jsonl"
    if suite_ok:
        write_prompt_file(cell_rows, cell_prompts)
        print(
            f"cell {CELL_C1_NATURAL['name']}: {len(cell_rows)} prompts (natural length)",
            flush=True,
        )
    else:
        # Still emit a blocked artifact: a missing suite is a recorded failure,
        # not a silent early exit.
        print("no cell prompts; emitting a blocked artifact", flush=True)

    arm_failures: dict[str, str] = {}
    # Results are keyed by (rep, arm) immediately. Compressing successes into a
    # per-arm list would shift a later rep's result into an earlier failed rep.
    runs: dict[tuple[int, str], dict[str, Any]] = {}

    def _attempt(rep: int, arm: str) -> None:
        try:
            runs[(rep, arm)] = run_arm(
                arm,
                model=args.model,
                prompts=cell_prompts,
                cell=CELL_C1_NATURAL,
                workdir=args.workdir,
                reps=1,
                timeout_s=args.arm_timeout,
            )
        except (MatchedBaselineError, subprocess.SubprocessError) as error:
            key = f"rep{rep}:{arm}"
            arm_failures[key] = f"{type(error).__name__}: {error}"
            print(f"rep {rep} arm {arm} FAILED: {arm_failures[key]}", flush=True)

    # Latin-square rotation over the three arms: every arm occupies every
    # position exactly once across three reps, so no arm is always first.
    latin = [
        ["tp1-d0", "tp2", "tp1-d1"],
        ["tp2", "tp1-d1", "tp1-d0"],
        ["tp1-d1", "tp1-d0", "tp2"],
    ]
    declared_orders: dict[int, list[str]] = {}
    if suite_ok:
        for rep in range(args.reps):
            order = latin[rep % len(latin)]
            declared_orders[rep] = order
            print(f"rep {rep}: order {order}", flush=True)
            for arm in order:
                _attempt(rep, arm)

    reps: list[dict[str, Any]] = []
    seen_run_ids: dict[str, str] = {}
    for rep in range(args.reps):
        present: dict[str, list[dict[str, Any]]] = {}
        run_ids: dict[str, str] = {}
        arm_entries: dict[str, Any] = {}
        for arm in ARMS:
            run = runs.get((rep, arm))
            if run is None:
                continue
            rows = (
                run.get("sweeps", [[]])[0]
                if arm == "tp2"
                else run.get("prompt_metrics", [])
            )
            run_ids[arm] = run["run_id"]
            # Validate BEFORE aggregation so a missing/None/NaN field or a
            # wrong row set produces a fail-closed artifact instead of raising.
            arm_row_failures = validate_arm_rows(
                arm, rows, expected_ids=expected_ids, expected_categories=expected_categories
            )
            for row in rows:
                arm_row_failures.extend(validate_accounting(arm, row, cell=CELL_C1_NATURAL))
            for failure in arm_row_failures:
                gate_failures.append(f"rep{rep} {failure}")
                accounting_failures.append(f"rep{rep} {failure}")
            entry: dict[str, Any] = {
                "run_id": run["run_id"],
                "command": run.get("command"),
                "log": run.get("log"),
                "device": run.get("device"),
                "devices": run.get("devices"),
                "route": run.get("route"),
                "session_capture_ms": run.get("session_capture_ms"),
                "session_destroy_ms": run.get("session_destroy_ms"),
                "tp1_provenance": run.get("tp1_provenance"),
                "tp1_timing_protocol": run.get("tp1_timing_protocol"),
                "prompt_metrics": rows,
            }
            if not arm_row_failures and rows:
                try:
                    entry.update(aggregate_rows(rows, arm=arm))
                    entry["categories"] = category_rows(rows, arm=arm)
                except (MatchedBaselineError, KeyError, TypeError, ValueError) as error:
                    failure = (
                        f"rep{rep} {arm}: aggregation failed: {type(error).__name__}: {error}"
                    )
                    gate_failures.append(failure)
                    accounting_failures.append(failure)
            arm_entries[arm] = entry
            present[arm] = rows
            if arm in TP1_ARMS and not run.get("tp1_provenance"):
                provenance_gaps.append(
                    f"rep{rep} {arm}: TP1 physical identity/profile/KV/variant provenance "
                    "not collected"
                )
        rep_entry: dict[str, Any] = {
            "rep": rep,
            "independent": True,
            "declared_order": declared_orders.get(rep, list(ARMS)),
            "run_ids": run_ids,
            "complete": all(arm in present for arm in ARMS),
            "arms": arm_entries,
        }
        rep_failures = validate_rep(rep_entry, seen_run_ids=seen_run_ids)
        gate_failures.extend(rep_failures)
        accounting_failures.extend(rep_failures)
        if rep_entry["complete"]:
            matching_failures.extend(
                f"rep{rep} {failure}"
                for failure in cross_arm_token_hashes(present, expected_ids=expected_ids)
            )
        reps.append(rep_entry)

    missing_arms = sorted(
        {
            arm
            for arm in ARMS
            if any((rep, arm) not in runs for rep in range(args.reps))
        }
    )
    if missing_arms:
        failure = f"missing arms (blocked): {missing_arms}"
        gate_failures.append(failure)
        accounting_failures.append(failure)
    gate_failures.extend(provenance_gaps)

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
    # Ratios are suppressed whenever input matching or provenance is unverified,
    # not only on per-row accounting failure.
    unverified = bool(accounting_failures or matching_failures or provenance_gaps)
    ratios = None if unverified else compute_ratios(reps)

    run_provenance = [
        {
            "rep": rep,
            "arm": arm,
            "run_id": run["run_id"],
            "command": run.get("command"),
            "log": run.get("log"),
            "device": run.get("device"),
            "devices": run.get("devices"),
            "route": run.get("route"),
            "session_capture_ms": run.get("session_capture_ms"),
            "session_destroy_ms": run.get("session_destroy_ms"),
            "tp1_provenance": run.get("tp1_provenance"),
            "tp1_timing_protocol": run.get("tp1_timing_protocol"),
        }
        for (rep, arm), run in sorted(runs.items())
    ]
    provenance: dict[str, Any] = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_revision": _git_revision(),
        "source_dirty": _git_dirty(),
        "host": _host_identity(),
        "model": {
            "path": str(args.model),
            "sha256": _model_sha256(args.model),
        },
        "runs": run_provenance,
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
            "destruction reported separately, Latin-square arm rotation"
        ),
        "window_overhead_note": (
            "Diagnostic windows carry a small unequal overhead: TP1 checks logit "
            "finiteness inside its timed loop, while TP2 performs its per-step "
            "argmax/list work inside the session call and its final np.stack/argmax "
            "after the timer. The rates are therefore a diagnostic comparison, not "
            "a product baseline."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_suite": {
            "canonical": str(args.prompts),
            "heldout": str(args.heldout_prompts),
            "prompts": len(cell_rows),
            "prompt_ids": expected_ids,
            "prompt_ids_sha256": prompt_ids_sha256(expected_ids),
            "categories": expected_categories,
        },
        "reps": reps,
        "ratios": ratios,
        "qualification": qualification,
        "qualified": all(qualification.values()) and all_gates_passed,
        "arm_failures": arm_failures,
        "missing_arms": missing_arms,
        "all_gates_passed": all_gates_passed,
        "accounting_failures": accounting_failures,
        "matching_failures": matching_failures,
        "provenance_gaps": provenance_gaps,
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
    return 0 if not unverified else 1


PRODUCT_ORDERS = (('tp1-d0', 'tp2', 'tp1-d1'), ('tp2', 'tp1-d1', 'tp1-d0'), ('tp1-d1', 'tp1-d0', 'tp2'))


def product_inputs(model, rows):
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(model))
    return [build_chat_prompt(tokenizer, str(row['prompt'])) for row in rows]


def product_identity(model):
    from scripts.tp2_resident_control import bind_resident_profile
    os.environ['HIPENGINE_GGUF_DECODE_REPACK'] = '1'
    profile = bind_resident_profile('production')
    paths = ('hipengine/core/memory.py', 'hipengine/runtime/qwen35_gguf_runner.py',
             'hipengine/runtime/gguf_decode_graph.py', 'hipengine/runtime/gguf_linear.py',
             'hipengine/distributed/tp2_generate.py', 'hipengine/distributed/shard_exec.py',
             'scripts/tp2_resident_control.py')
    return {'model_sha256': _model_sha256(model), 'source_revision': _git_revision(),
        'staged_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--cached', '--binary'])).hexdigest(),
        'source_sha256': {p: hashlib.sha256((REPO_ROOT/p).read_bytes()).hexdigest() for p in paths},
        'profile_sha256': profile['manifest_sha256'], 'capacity': 200, 'kv': 'bf16',
        'recurrent': 'fp32', 'sampling': 'greedy', 'host': _host_identity()}


def _product_token(token, vocab):
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)) or not 0 <= token < vocab:
        raise ValueError(f'invalid product sample {token!r}')
    return int(token)


def measure_product_prompt(adapter, tokens, prompt_row, *, clock=time.perf_counter, transitions=128):
    """Identical outer clocks, native asymmetric transfers; checks after timers."""
    from scripts.tp2_xtx_tp1_eager_stage_probe import validate_result
    total_start = clock()
    start = clock()
    first = token = _product_token(adapter.prefill(tokens), adapter.vocab_size)
    prefill_ms = (clock() - start) * 1000
    start = clock()
    adapter.begin_decode(transitions)
    capture_ms = (clock() - start) * 1000
    outputs = []
    start = clock()
    for step in range(transitions):
        result = adapter.transition(token, return_logits=(not adapter.resident or step == transitions-1))
        token = _product_token(result.token_id, adapter.vocab_size)
        outputs.append(token)
    decode_ms = (clock() - start) * 1000
    start = clock()
    adapter.end_decode()
    destroy_ms = (clock() - start) * 1000
    total_ms = (clock() - total_start) * 1000
    validate_result(result, adapter.vocab_size)  # No extra quality scan inside timing.
    if adapter.position != len(tokens) + transitions:
        raise ValueError('product transition count/cursor mismatch')
    return {'id': prompt_row['id'], 'category': prompt_row['category'],
        'prompt_tokens': len(tokens), 'prompt_token_sha256': token_tuple_sha256(tokens),
        'timed_decode_transitions': transitions, 'total_samples': transitions+1,
        'user_visible_requested_horizon': transitions, 'api_128_completion_equivalent': False,
        'decode_position_start': len(tokens), 'decode_position_end': len(tokens)+transitions-1,
        'prefill_sample_id': first, 'sampled_output_ids': outputs, 'sampled_output_sha256': _ids_hash(outputs),
        'vocab_size': adapter.vocab_size, 'finite_final_logits': True,
        'finite_verification': 'final-only in timed run; separate all-position correctness companion',
        'native_full_logits_every_step': not adapter.resident, 'graph_effective': True,
        'prefill_ms': prefill_ms, 'capture_ms': capture_ms, 'decode_ms': decode_ms,
        'destroy_ms': destroy_ms, 'total_generation_ms': total_ms}


def product_summary(rows):
    transitions = sum(r['timed_decode_transitions'] for r in rows)
    samples = sum(r['total_samples'] for r in rows)
    totals = {k: sum(r[k] for r in rows) for k in ('decode_ms', 'total_generation_ms', 'prefill_ms', 'capture_ms', 'destroy_ms')}
    if not rows or any(not math.isfinite(v) or v < 0 for v in totals.values()) or totals['decode_ms'] <= 0 or totals['total_generation_ms'] <= 0:
        raise ValueError('invalid product timing denominator')
    return {**totals, 'decode_transitions': transitions, 'total_samples': samples,
            'decode_tok_s': transitions * 1000 / totals['decode_ms'],
            'generation_samples_s': samples * 1000 / totals['total_generation_ms']}


def validate_product_run(run, *, arm, rep, expected_identity, rows, token_rows):
    if run.get('status') != 'complete' or run.get('natural_teardown') is not True or run.get('first_bad_stage') is not None:
        raise ValueError('product arm failed or did not close naturally')
    if run.get('arm') != arm or run.get('rep') != rep or not run.get('run_id'):
        raise ValueError('stale/mismatched execution identity')
    if not expected_identity or run.get('identity') != expected_identity:
        raise ValueError('product provenance mismatch/missing evidence')
    for key in ('model_sha256', 'source_revision', 'source_sha256', 'profile_sha256', 'staged_diff_sha256'):
        if not expected_identity.get(key): raise ValueError(f'missing identity field {key}')
    if run.get('profile',{}).get('manifest_sha256') != expected_identity['profile_sha256']:
        raise ValueError('missing/mismatched bound profile')
    if run.get('route',{}).get('max_sequence_length') != 200:
        raise ValueError('missing/mismatched actual route capacity')
    scope = run.get('scope_manifest',{})
    if not scope.get('manifest') or _sha256_json(scope['manifest']) != scope.get('sha256'):
        raise ValueError('missing/invalid actual scope manifest')
    for key in ('session_capture_ms','session_graph_destroy_ms','session_teardown_ms'):
        value = _as_float(run.get(key))
        if value is None or value < 0: raise ValueError(f'missing/invalid {key}')
    failures = validate_arm_rows(arm, run.get('rows', []), expected_ids=[r['id'] for r in rows],
                                expected_categories={r['id']: r['category'] for r in rows})
    if failures: raise ValueError(str(failures))
    for r, tokens in zip(run['rows'], token_rows, strict=True):
        if r['prompt_token_sha256'] != token_tuple_sha256(tokens) or r['prompt_tokens'] != len(tokens):
            raise ValueError('product input hash/length mismatch')
        for key, expected in [('timed_decode_transitions',128), ('total_samples',129), ('user_visible_requested_horizon',128),
                              ('decode_position_start',len(tokens)), ('decode_position_end',len(tokens)+127)]:
            if type(r.get(key)) is not int or r[key] != expected: raise ValueError(f'bad {key}')
        if r.get('api_128_completion_equivalent') is not False or r.get('graph_effective') is not True:
            raise ValueError('wrong product horizon/route')
        if r.get('native_full_logits_every_step') is not (arm == 'tp2'):
            raise ValueError('native readback protocol mismatch')
        if r.get('finite_final_logits') is not True: raise ValueError('nonfinite final output')
        ids = r.get('sampled_output_ids', [])
        if len(ids) != 128 or r.get('sampled_output_sha256') != _ids_hash(ids): raise ValueError('sample hash/count mismatch')
        for token in [r['prefill_sample_id'], *ids]: _product_token(token, r['vocab_size'])
        for key in ('decode_ms','total_generation_ms','prefill_ms','capture_ms','destroy_ms'):
            value = _as_float(r.get(key))
            if value is None or value < 0 or (key in ('decode_ms','total_generation_ms') and value <= 0):
                raise ValueError(f'invalid {key}')
        if r['total_generation_ms'] + 1e-6 < sum(r[k] for k in ('decode_ms','prefill_ms','capture_ms','destroy_ms')):
            raise ValueError('outer generation window does not contain its phases')


def product_idle_gate():
    from scripts.tp2_teacher_bisect_parent import wait_for_idle
    idle, snapshot = wait_for_idle(max_load=2.0, max_gpu=5.0, timeout_s=180, poll_s=5)
    if not idle: raise ValueError(f'idle gate failed: {snapshot}')
    return {'idle': True, **snapshot}


def product_system_snapshot():
    snapshot = {}
    for name, command in [('hipcc',['hipcc','--version']),
        ('gpu',['rocm-smi','--showuse','--showtemp','--showpower','--showclocks','--showmemuse','--json'])]:
        try:
            result = subprocess.run(command,capture_output=True,text=True,timeout=20)
            snapshot[name] = {'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        except (OSError,subprocess.SubprocessError) as error:
            snapshot[name] = {'error':str(error)}
    return snapshot


def product_worker(args):
    from scripts.tp2_resident_control import create_native_adapter, bind_resident_profile, resolved_scope_manifest
    from scripts.tp2_xtx_tp1_eager_stage_probe import StageRecorder
    arm = args.product_arm
    visibility = os.environ.get('HIP_VISIBLE_DEVICES')
    if (arm == 'tp2' and visibility) or (arm != 'tp2' and visibility != str(TP1_ARMS.index(arm))):
        raise ValueError('product worker physical visibility mismatch')
    rows = load_prompt_rows(args.prompts)
    token_rows = product_inputs(args.model, rows)
    identity = product_identity(args.model)
    if max(map(len, token_rows)) + 129 > 200: raise ValueError('product capacity exceeded')
    record = StageRecorder(args.json, {'kind':'tp2_product_arm', 'arm':arm, 'rep':args.rep_index,
        'run_id':uuid.uuid4().hex, 'identity':identity, 'profile':bind_resident_profile('production'),
        'command':shlex.join([sys.executable,*sys.argv]), 'git_status':subprocess.check_output(['git','status','--short'],text=True),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows':[]})
    state = {}
    def build():
        t = time.perf_counter()
        state['adapter'] = a = create_native_adapter(args.model, arm, capacity=200)
        record.artifact['devices'] = _device_identities(a.owner)
        record.artifact['route'] = _resolved_route(a.owner)
        return {'model_load_ms':(time.perf_counter()-t)*1000}
    def timed_guard(stage, fn, field):
        def timed():
            start = time.perf_counter()
            fn()
            record.artifact[field] = (time.perf_counter()-start)*1000
            return {field:record.artifact[field]}
        return record.guard(stage,timed)
    if record.guard('build', build):
        a = state['adapter']
        timed_guard('session-graph-capture', a.prepare, 'session_capture_ms')
        if not record.exit_code:
            record.artifact['scope_manifest'] = resolved_scope_manifest(a.owner)
            ranks = record.artifact['scope_manifest']['manifest']['ranks']
            kv = ([a.session.kv_storage_dtype] if a.resident else [s.kv_storage_dtype for s in a.session._scratches.values()])
            record.artifact['effective'] = {'kv': [str(v) for v in kv],
                'recurrent_fp16': [r['fp16_recurrent_state'] for r in ranks.values()],
                'timed_decode_route': 'state_bound_hipgraph' if a.resident else 'per_layer_hipgraph_native_exchange',
                'sampling': 'greedy', 'eos': None, 'request_warmup_transitions': 0,
                'global_warmup': {'prompt_tokens':16,'transitions':2},
                'readback': 'token_only_except_final' if a.resident else 'native_full_logits_every_transition'}
            if any(r['fp16_recurrent_state'] for r in ranks.values()) or any(str(v).lower().split('.')[-1]!='bf16' for v in kv):
                raise ValueError('actual state/KV precision differs from product declaration')
            timed_guard('warmup', lambda: measure_product_prompt(a, token_rows[0][:16], rows[0], transitions=2), 'warmup_ms')
        if not args.capacity_only and not record.exit_code:
            record.artifact['system_before'] = product_system_snapshot()
            for row,tokens in zip(rows,token_rows,strict=True):
                def measure():
                    metric = measure_product_prompt(a,tokens,row)
                    if not a.resident:
                        metric['native_trace_decode_ms'] = {key:1000*sum(t.stages.get(key,0) for t in a.traces if t.kind=='decode')
                            for key in {k for t in a.traces for k in t.stages}}
                        metric['trace_note'] = 'Host stage attribution only; never the throughput denominator.'
                    record.artifact['rows'].append(metric)
                    return metric
                if not record.guard(row['id'],measure): break
        if not record.exit_code:
            if not args.capacity_only: record.artifact['system_after'] = product_system_snapshot()
            timed_guard('session-graph-destroy', a.destroy_graphs, 'session_graph_destroy_ms')
            timed_guard('teardown', a.close, 'session_teardown_ms')
    record.artifact['capacity_only'] = bool(args.capacity_only)
    record.artifact['natural_teardown'] = not bool(record.exit_code)
    record.finish()
    if record.exit_code:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    return 0


def run_product_child(arm, rep, args, prompts):
    run_id = uuid.uuid4().hex
    output = args.workdir / f'product-r{rep}-{arm}-{run_id}.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--protocol','product', '--product-arm',arm,
               '--rep-index',str(rep), '--model',str(args.model), '--prompts',str(prompts), '--json',str(output)]
    env = dict(os.environ)
    env.pop('ROCR_VISIBLE_DEVICES',None)
    if arm == 'tp2': env.pop('HIP_VISIBLE_DEVICES',None)
    else: env['HIP_VISIBLE_DEVICES'] = str(TP1_ARMS.index(arm))
    log = output.with_suffix('.log')
    with log.open('w') as handle:
        result = subprocess.run(['timeout','-k','10s',str(args.arm_timeout)+'s',*command], env=env,
                                stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode != 0 or not output.exists():
        raise ValueError(f'{arm} failed rc={result.returncode}; log={log}')
    payload = json.loads(output.read_text())
    payload['outer_command'] = shlex.join(command)
    payload['log'] = str(log)
    return payload


def run_product_campaign(args):
    from scripts.tp2_xtx_tp1_eager_stage_probe import StageRecorder
    args.workdir = args.workdir or Path('/tmp/tp2-product-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    args.workdir.mkdir(parents=True,exist_ok=True)
    rows, failures = load_cell_prompts(args.prompts,args.heldout_prompts)
    tokens = product_inputs(args.model,rows) if rows else []
    identity = product_identity(args.model)
    prompt_path = args.workdir/'product-prompts.jsonl'
    write_prompt_file(rows,prompt_path)
    record = StageRecorder(args.json, {'kind':'tp2_native_product_baseline','identity':identity,
        'protocol':'128 timed transitions at P..P+127; 129 total samples, not API max_new_tokens=128 latency',
        'transfer_asymmetry':'TP1 token-only except final logits; TP2 native full logits and host argmax',
        'qualified':False, 'performance_claim':'diagnostic measured baseline only', 'runs':[], 'ratios':None,
        'orders':[list(PRODUCT_ORDERS[r%3]) for r in range(args.reps)],
        'prompt_ids':[r['id'] for r in rows], 'prompt_token_hashes':[token_tuple_sha256(t) for t in tokens]})
    if failures or not rows: raise ValueError(f'bad suite: {failures}')
    def check_quality():
        quality = json.loads(Path(args.quality_json).read_text())
        if quality.get('all_gates_passed') is not True or quality.get('positions') != len(rows)*128:
            raise ValueError('missing/failed sustained D128 correctness companion')
        for key in ('model_sha256','source_sha256','profile_sha256','capacity','kv','recurrent','sampling','host'):
            if quality['identity'].get(key) != identity.get(key): raise ValueError(f'stale correctness {key}')
        if quality['suite']['ids'] != [r['id'] for r in rows] or quality['suite']['tokens'] != tokens:
            raise ValueError('correctness/product prompt mismatch')
        record.artifact['correctness'] = {'path':str(args.quality_json),
            'sha256':hashlib.sha256(Path(args.quality_json).read_bytes()).hexdigest(), 'positions':quality['positions']}
        return {'sustained_gate_passed':True}
    if not record.guard('correctness-companion',check_quality):
        record.finish(); return 1
    seen = set()
    for rep in range(args.reps):
        for arm in PRODUCT_ORDERS[rep%3]:
            def execute():
                idle = product_idle_gate()
                run = run_product_child(arm,rep,args,prompt_path)
                validate_product_run(run,arm=arm,rep=rep,expected_identity=identity,rows=rows,token_rows=tokens)
                if run['run_id'] in seen: raise ValueError('execution reused across reps')
                seen.add(run['run_id'])
                run['idle_gate'] = idle
                run['aggregate'] = product_summary(run['rows'])
                overhead = sum(run.get(k,0) for k in ('session_capture_ms','session_graph_destroy_ms','session_teardown_ms'))
                run['aggregate']['session_generation_ms_including_graphs_and_teardown'] = run['aggregate']['total_generation_ms'] + overhead
                run['aggregate']['session_generation_samples_s'] = 1000*run['aggregate']['total_samples']/(run['aggregate']['total_generation_ms']+overhead)
                run['categories'] = {c:product_summary([r for r in run['rows'] if r['category']==c]) for c in sorted({r['category'] for r in rows})}
                record.artifact['runs'].append(run)
                return {'run_id':run['run_id'],'decode_tok_s':run['aggregate']['decode_tok_s']}
            if not record.guard(f'rep-{rep}/{arm}',execute):
                record.finish()
                return 1
    def ratios():
        per_rep = []
        for rep in range(args.reps):
            runs = {r['arm']:r for r in record.artifact['runs'] if r['rep']==rep}
            d0 = runs['tp1-d0']['devices']['0']['uuid']; d1 = runs['tp1-d1']['devices']['0']['uuid']
            if d0==d1 or {d0,d1}!={d['uuid'] for d in runs['tp2']['devices'].values()}:
                raise ValueError('physical paired denominators mismatch')
            rates = {a:r['aggregate']['decode_tok_s'] for a,r in runs.items()}
            faster = max(TP1_ARMS,key=lambda a:rates[a])
            per_rep.append({'rep':rep,'rates':rates,'faster_tp1_arm':faster,'tp2_vs_faster_tp1':rates['tp2']/rates[faster]})
        record.artifact['ratios'] = {'per_rep':per_rep,
            'median_tp2_vs_faster_tp1':float(np.median([r['tp2_vs_faster_tp1'] for r in per_rep]))}
        return {'paired_reps':len(per_rep)}
    record.guard('paired-ratios',ratios)
    record.finish()
    return record.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
