#!/usr/bin/env python3
"""Section-6.1 teacher-forced gate: single-row AR versus multi-row verification.

The production AR route consumes one token at a time and predicts the next;
the production target verifier consumes a block of ``rows`` tokens at once and
predicts the same positions. Both run the shipped quant routes, so this gate
isolates verification-specific drift (row packing, block ownership, deferred
state) from quant-route drift.

For every fixture prompt the gate:

1. prefills, then walks the production AR route for ``max_budget + 1`` steps
   with full logits, recording the greedy trajectory and the per-position
   single-row logits;
2. for each candidate budget, resets, re-prefills, and runs the production
   native target block on the AR trajectory with full logits;
3. compares each verifier row against the AR logits for the same position.

Reported metrics are the section-6.1 screen on the position-pooled vector:
mean/p95/p99/max row KL, overall top-1, and top-1 per category and per
candidate budget. Generated-token agreement is recorded as a diagnostic.
The command writes its report and exits nonzero when any repeat fails the
screen, finiteness, requested native-route, or repeat-signature check.

CPU/GPU: needs a GPU. Raw logits never leave the process.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = (
    REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
    REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl",
)
# Section 6.1 calibrated production envelope.
ENVELOPE = {"mean": 1e-3, "p95": 5e-3, "p99": 2e-2, "max": 5e-2}
TOP1_OVERALL = 0.99
TOP1_PER_SCOPE = 0.97


def _row_kl(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Per-row KL(reference || candidate) over the last axis."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"logit shape mismatch: {ref.shape} vs {cand.shape}")
    ref = ref - ref.max(axis=-1, keepdims=True)
    cand = cand - cand.max(axis=-1, keepdims=True)
    log_p = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    log_q = cand - np.log(np.exp(cand).sum(axis=-1, keepdims=True))
    return np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)


def _load_prompts(paths) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _seed_ids(tokenizer, build_chat_prompt, row: dict) -> list[int]:
    """Tokenize the fixture instruction with the model's own chat template."""

    messages = row.get("messages") or []
    content = str(messages[0]["content"]) if messages else ""
    ids = list(build_chat_prompt(tokenizer, content))
    if not ids:
        raise RuntimeError(f"prompt {row.get('id')!r} tokenized to nothing")
    return ids


def _extend_to_window(session, seed_ids: list[int], prompt_tokens: int) -> list[int]:
    """Extend the real instruction window with the model's own continuation.

    The instruction content leads, so the prefill window crosses the >=129-row
    owner boundary while still being driven by the real fixture prompt.
    """

    ids = list(seed_ids)
    if len(ids) >= prompt_tokens:
        return ids[-prompt_tokens:]
    session.reset()
    cur = session.prefill(ids, use_bulk=True, bulk_attention_mode="bulk",
                          return_logits=True)
    while len(ids) < prompt_tokens:
        token = int(cur.token_id)
        ids.append(token)
        cur = session.step(token, return_logits=True)
    session.reset()
    return ids


def _run_prompt(session, row, ids, budgets, *, require_native: bool) -> dict:
    """AR teacher pass, then one native verifier block per candidate budget."""

    from hipengine.benchmark.correctness import evaluate_logits

    max_budget = max(budgets)
    session.reset()
    first = session.prefill(ids, use_bulk=True, bulk_attention_mode="bulk",
                            return_logits=True)
    ar_logits = [np.asarray(first.logits, dtype=np.float32).reshape(-1)]
    ar_tokens = [int(first.token_id)]
    # ar_logits[j] is the logits for the position that consumes ar_tokens[j-1]
    # (ar_logits[0] is the last prompt token). The verifier block starts at the
    # session cursor after prefill, so its row j corresponds to ar_logits[j+1],
    # and a B<max_budget> block needs ar_logits up to index max_budget + 1.
    for _ in range(max_budget + 1):
        step = session.step(ar_tokens[-1], return_logits=True)
        ar_logits.append(np.asarray(step.logits, dtype=np.float32).reshape(-1))
        ar_tokens.append(int(step.token_id))
    ar_logits = np.stack(ar_logits, axis=0)  # rows 0..max_budget+1

    cases = []
    for budget in budgets:
        rows = budget + 1
        session.reset()
        session.prefill(ids, use_bulk=True, bulk_attention_mode="bulk",
                        return_logits=False)
        root_position = int(session.position)
        result = session.verify_target_block_native_cycle(
            ar_tokens[:rows],
            fallback=not require_native,
            cycle_id=0,
            transaction_id=1,
            request_id=1,
            device_accept_commit=True,
            remaining_decode=budget,
            bulk_attention_mode="native",
            capture_linear_state_rows=True,
            defer_linear_state_commit=True,
            capture_lm_head_logits=True,
        )
        fallback_reason = getattr(
            session, "last_native_spec_target_fallback_reason", None)
        logits = result.lm_head_logits_f32
        if logits is None:
            raise RuntimeError(
                f"budget {budget} returned no lm_head logits; the native "
                "verifier did not capture them")
        verifier = np.ascontiguousarray(logits, dtype=np.float32).reshape(rows, -1)
        teacher = ar_logits[1:rows + 1]
        if verifier.shape != teacher.shape:
            raise RuntimeError(
                f"budget {budget} row shape mismatch: verifier {verifier.shape} "
                f"vs teacher {teacher.shape}")
        kl = _row_kl(teacher, verifier)
        top1 = (np.argmax(teacher, axis=-1) == np.argmax(verifier, axis=-1))
        metrics = evaluate_logits(teacher, verifier)
        cases.append({
            "budget": int(budget),
            "rows": int(rows),
            "root_position": root_position,
            "native_graph": fallback_reason is None,
            "fallback_reason": fallback_reason,
            "target_top1": [int(v) for v in getattr(result, "target_top1", ())],
            "ar_top1": [int(v) for v in ar_tokens[1:rows + 1]],
            "target_top1_matches_ar": bool(
                list(getattr(result, "target_top1", ()))
                == list(ar_tokens[1:rows + 1])),
            "kl": [float(v) for v in kl],
            "kl_mean": float(np.mean(kl)),
            "kl_max": float(np.max(kl)),
            "top1": [bool(v) for v in top1],
            "top1_agreement": float(np.mean(top1)),
            "max_abs_diff": float(np.max(np.abs(teacher - verifier))),
            "finite": bool(np.isfinite(verifier).all()),
            "evaluate_logits_kl_mean": float(metrics.kl_mean),
            "evaluate_logits_kl_max": float(metrics.kl_max),
            "evaluate_logits_top1": float(metrics.top1_agreement),
        })
    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "prompt_tokens": len(ids),
        "ar_tokens": ar_tokens,
        "cases": cases,
    }


def _pool(results: list[dict]) -> dict:
    kl: list[float] = []
    top1: list[bool] = []
    by_scope: dict[str, list[bool]] = defaultdict(list)
    by_budget: dict[str, list[bool]] = defaultdict(list)
    kl_by_budget: dict[str, list[float]] = defaultdict(list)
    above_p99: list[dict] = []
    for row in results:
        for case in row["cases"]:
            kl.extend(case["kl"])
            top1.extend(case["top1"])
            by_scope[str(row["category"])].extend(case["top1"])
            by_budget[str(case["budget"])].extend(case["top1"])
            kl_by_budget[str(case["budget"])].extend(case["kl"])
    arr = np.asarray(kl, dtype=np.float64)
    p99 = float(np.percentile(arr, 99)) if arr.size else 0.0
    for row in results:
        for case in row["cases"]:
            for index, value in enumerate(case["kl"]):
                if value > p99:
                    above_p99.append({
                        "id": row.get("id"),
                        "category": row.get("category"),
                        "budget": case["budget"],
                        "row": index,
                        "kl": float(value),
                    })
    return {
        "rows": int(arr.size),
        "kl_mean": float(arr.mean()) if arr.size else 0.0,
        "kl_p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "kl_p99": p99,
        "kl_max": float(arr.max()) if arr.size else 0.0,
        "top1_agreement": float(np.mean(top1)) if top1 else 0.0,
        "top1_by_scope": {
            scope: {"rows": len(values), "agreement": float(np.mean(values))}
            for scope, values in sorted(by_scope.items())
        },
        "top1_by_budget": {
            budget: {"rows": len(values), "agreement": float(np.mean(values))}
            for budget, values in sorted(by_budget.items())
        },
        "kl_by_budget": {
            budget: {
                "rows": len(values),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
            for budget, values in sorted(kl_by_budget.items())
        },
        "rows_above_p99": above_p99,
    }


def _screen(pooled: dict) -> dict:
    checks = {
        "mean_kl": pooled["kl_mean"] <= ENVELOPE["mean"],
        "p95_kl": pooled["kl_p95"] <= ENVELOPE["p95"],
        "p99_kl": pooled["kl_p99"] <= ENVELOPE["p99"],
        "max_kl": pooled["kl_max"] <= ENVELOPE["max"],
        "top1_overall": pooled["top1_agreement"] >= TOP1_OVERALL,
        "top1_per_scope": all(
            value["agreement"] >= TOP1_PER_SCOPE
            for value in pooled["top1_by_scope"].values()
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, nargs="*", default=list(DEFAULT_PROMPTS))
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--budgets", default="1,2,3")
    ap.add_argument("--repeat-runs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-sequence-length", type=int, default=1024)
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--quant", default=None,
                    help="prefill quant axis for the warmup prefill")
    ap.add_argument("--allow-eager-fallback", action="store_true",
                    help="permit a non-native verifier fallback (diagnostic)")
    ap.add_argument("--require-native-graph", action="store_true",
                    help="fail unless every case used the captured native graph")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    if args.allow_eager_fallback and args.require_native_graph:
        raise SystemExit("--allow-eager-fallback and --require-native-graph conflict")
    if args.repeat_runs < 1:
        ap.error("--repeat-runs must be positive")
    if args.prompt_tokens < 1:
        ap.error("--prompt-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    budgets = tuple(int(v) for v in str(args.budgets).split(",") if v.strip())
    if not budgets or any(b < 1 for b in budgets):
        raise SystemExit("--budgets must be a CSV of positive integers")

    prompt_rows = _load_prompts(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: int(args.limit)]
    if not prompt_rows:
        ap.error("at least one prompt is required")

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from scripts.gguf_mtp_bench import build_chat_prompt

    compiler_version = (Path(args.compiler_version_file).read_text()
                        if args.compiler_version_file else None)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))

    runs: list[dict] = []
    with Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        max_sequence_length=int(args.max_sequence_length),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        if args.quant:
            session.select_prefill_quant(str(args.quant))
        for repeat in range(int(args.repeat_runs)):
            results = []
            for row in prompt_rows:
                seed = _seed_ids(tokenizer, build_chat_prompt, row)
                ids = _extend_to_window(session, seed, int(args.prompt_tokens))
                results.append(_run_prompt(
                    session, row, ids, budgets,
                    require_native=bool(args.require_native_graph)))
            pooled = _pool(results)
            runs.append({
                "repeat": repeat,
                "pooled": pooled,
                "screen": _screen(pooled),
                "results": results,
            })

    signatures = []
    for run in runs:
        signature = [
            [[case["kl"], case["top1"]] for case in row["cases"]]
            for row in run["results"]
        ]
        signatures.append(json.dumps(signature, sort_keys=True))
    deterministic = len(set(signatures)) == 1
    cases = [case for run in runs for row in run["results"] for case in row["cases"]]
    checks = {
        "nonempty_results": bool(cases) and all(run["pooled"]["rows"] > 0 for run in runs),
        "numerical_envelope": all(run["screen"]["passed"] for run in runs),
        "deterministic_repeats": bool(deterministic),
        "finite_logits": all(case["finite"] for case in cases),
        "native_graph": not args.require_native_graph or all(
            case["native_graph"] and case["fallback_reason"] is None for case in cases
        ),
    }
    passed = all(checks.values())
    payload = {
        "schema": "hipengine.ud_mtp_ar_verify_numerics_gate.v1",
        "kind": "correctness_gate",
        "performance_claim": False,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "plan": "docs/campaigns/UD-GFX1151-OPTIMIZE.md",
        "phase": "Phase 3: teacher-forced single-row AR versus multi-row verification",
        "model": str(args.model),
        "quant_axis": args.quant,
        "protocol": {
            "teacher": "production single-row AR route (session.step, return_logits)",
            "candidate": "production native target block (device accept/commit, "
                         "bulk_attention_mode=native, remaining_decode=budget)",
            "prompts": [str(p) for p in args.prompts],
            "prompt_tokens": int(args.prompt_tokens),
            "budgets": list(budgets),
            "repeat_runs": int(args.repeat_runs),
            "envelope": ENVELOPE,
            "top1_overall": TOP1_OVERALL,
            "top1_per_scope": TOP1_PER_SCOPE,
        },
        "deterministic": bool(deterministic),
        "checks": checks,
        "passed": passed,
        "runs": runs,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    pooled = runs[0]["pooled"]
    print(f"[ar-verify] rows={pooled['rows']} mean={pooled['kl_mean']:.3e} "
          f"p95={pooled['kl_p95']:.3e} p99={pooled['kl_p99']:.3e} "
          f"max={pooled['kl_max']:.3e} top1={pooled['top1_agreement']:.4f} "
          f"passed={passed} deterministic={deterministic}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
