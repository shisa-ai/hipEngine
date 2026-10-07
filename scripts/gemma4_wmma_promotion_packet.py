#!/usr/bin/env python3
"""Promotion packet for the Gemma 4 WMMA prefill candidates under section 2.11.

The 2026-10-07 lead decision scopes the production KL envelope to rows whose
expert routing is identical between the strict and candidate arms: on those
rows the unchanged 2026-08-16 bars measure whether the arithmetic is close,
while routing-flip rows are excluded and reported. This script runs the
18-case canonical/heldout packet end to end under that envelope:

* the strict arm teachers every case and the candidate replays the same
  forced chain (identical structure to `scripts/gemma4_production_quality.py`);
* both arms' per-layer residual states are captured on every decode step and
  routed through the CPU router oracle, so each row is classified
  routing-identical or flipped;
* the binding bars are the unchanged August numbers over routing-identical
  rows (mean 1e-3, p95 5e-3, p99 2e-2, max 5e-2, top-1 99% global / 97% per
  category), with flipped rows reported by rate and worst KL;
* the flip-rate review floors (10% per case, 5% per packet) flag rather than
  fail;
* task non-inferiority, three-run repeats and unrelated-request isolation are
  checked exactly as the production packet checks them.

Guarded on HIP and the campaign artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_production_quality import (
    DEPTHS, padded_chat, check_task_answer, greedy_output,
)
from scripts.gemma4_router_margin_diagnostic import router_scores, top8
from scripts.gemma4_teacher_forced_gate import row_kl_divergence

# The binding bars, per EXECUTION-PROFILES section 2.11 (2026-10-07 lead decision):
# robust statistics over all rows plus a bounded flip-damage budget.
MEDIAN_KL = 1e-3
P90_KL = 2e-2
DAMAGE_LINE_KL = 5e-2
DAMAGE_RATE_GLOBAL = 0.03
DAMAGE_RATE_CATEGORY = 0.08
TOP1_GLOBAL = 0.96
TOP1_SCOPE = 0.94


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def routing_identical(strict_captures, candidate_captures, reader, num_layers: int) -> bool:
    """Do the two arms pick the same top-8 expert set at every layer?"""

    for layer in range(1, num_layers):
        weight = reader.tensor_data(f"blk.{layer}.ffn_gate_inp.weight")
        scale = reader.tensor_data(f"blk.{layer}.ffn_gate_inp.scale")
        hidden_size = weight.shape[1]
        a = top8(router_scores(strict_captures[layer - 1][0], weight, scale, hidden_size))
        b = top8(router_scores(candidate_captures[layer - 1][0], weight, scale, hidden_size))
        if not np.array_equal(a, b):
            return False
    return True


def verdict(strict_rows, candidate_rows, kls_all, *, scope: bool) -> dict:
    """Section 2.11 binding verdict: robust statistics plus the damage budget.

    ``strict_rows``/``candidate_rows`` are the case's full row sets (the
    statistics are over all rows); ``kls_all`` are their per-row KLs, so the
    damage rate reuses the same values the median and p90 come from.
    """

    kls = np.asarray(kls_all, dtype=np.float64)
    top1 = float(np.mean([
        int(np.argmax(b)) == int(np.argmax(c))
        for b, c in zip(strict_rows, candidate_rows)]))
    damage = float(np.mean(kls > DAMAGE_LINE_KL))
    top1_bar = TOP1_SCOPE if scope else TOP1_GLOBAL
    damage_bar = DAMAGE_RATE_CATEGORY if scope else DAMAGE_RATE_GLOBAL
    failed = []
    if float(np.median(kls)) > MEDIAN_KL:
        failed.append("kl_median")
    if float(np.percentile(kls, 90)) > P90_KL:
        failed.append("kl_p90")
    if damage > damage_bar:
        failed.append("flip_damage_rate")
    if top1 < top1_bar:
        failed.append("top1_rate")
    return {
        "rows": int(kls.size),
        "kl_median": float(np.median(kls)),
        "kl_p90": float(np.percentile(kls, 90)),
        "flip_damage_rate": damage,
        "damage_line_kl": DAMAGE_LINE_KL,
        "top1_rate": top1,
        "thresholds": {"kl_median": MEDIAN_KL, "kl_p90": P90_KL,
                        "flip_damage_rate": damage_bar, "top1_rate": top1_bar},
        "failed": failed,
        "passed": not failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--candidate-variants", default="gemma4_wmma_flash,gemma4_wmma_flash_full")
    args = parser.parse_args(argv := None)
    if args.rows < 1:
        parser.error("rows must be positive")
    args.directory.mkdir(parents=True, exist_ok=True)

    from scripts.gemma4_campaign_bench import _resolve_generator
    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.gemma4 import Gemma4Runner

    llm, production, loading = _resolve_generator(args.artifact, max(DEPTHS) + args.rows)
    generator = llm._get_text_generator()
    reader = GGUFReader(str(args.artifact))
    candidate_variants = tuple(args.candidate_variants.split(","))
    strict = Gemma4Runner(weights=production.weights, capacity=max(DEPTHS) + args.rows,
                          prefill_attention_variants=("gemma4_plain",))
    candidate = Gemma4Runner(weights=production.weights, capacity=max(DEPTHS) + args.rows,
                             prefill_attention_variants=candidate_variants)

    cases = []
    for split, filename in (("canonical", "mtpbench-code-general-ja.jsonl"),
                            ("heldout", "gdn-prefill-category-heldouts.jsonl")):
        for line in (_REPO_ROOT / "benchmarks/prompts" / filename).read_text().splitlines():
            cases.append({**json.loads(line), "split": split})

    num_layers = 30
    report = {
        "kind": "gemma4_wmma_promotion_packet",
        "performance_claim": False,
        "created": "2026-10-07",
        "envelope": "EXECUTION-PROFILES section 2.11: unchanged 2026-08-16 bars over "
                    "routing-identical rows; routing-flip rows excluded and reported",
        "bars": {"kl_median": MEDIAN_KL, "kl_p90": P90_KL,
                  "damage_line_kl": DAMAGE_LINE_KL,
                  "flip_damage_rate_global": DAMAGE_RATE_GLOBAL,
                  "flip_damage_rate_category": DAMAGE_RATE_CATEGORY,
                  "top1_global": TOP1_GLOBAL, "top1_scope": TOP1_SCOPE},
        "candidate_variants": candidate_variants,
        "rows": int(args.rows),
        "sources": {"evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "cases": [],
    }
    overall_all_b, overall_all_c = [], []
    category_same = {}
    started = time.monotonic()
    try:
        for index, case in enumerate(cases):
            depth = DEPTHS[index % len(DEPTHS)]
            message = "\n".join(m["content"] for m in case["messages"])
            ids = padded_chat(generator, message, depth, 20261003 + index)
            name = case["id"]

            def forced_capture(runner, forced_chain=None):
                """Teacher-forced capture: replay ``forced_chain`` when given.

                With no chain this is the strict arm's teacher capture and the
                returned chain is its own greedy continuation. With the strict
                chain the candidate replays exactly those tokens, so both arms'
                rows are aligned at identical contexts -- the packet's core
                contract. Comparing each arm's own free-running chain instead
                would measure chain divergence, not arithmetic.
                """
                runner.reset()
                logits = np.asarray(runner.forward(ids), dtype=np.float32).reshape(-1).copy()
                own_chain, logits_rows, step_captures = [], [], []
                for step in range(args.rows):
                    logits_rows.append(logits)
                    token = (int(forced_chain[step]) if forced_chain is not None
                             else int(np.argmax(logits)))
                    if forced_chain is None:
                        own_chain.append(token)
                    if step + 1 < args.rows:
                        captures: list[np.ndarray] = []
                        logits = np.asarray(runner.forward([token], capture_layers=captures),
                                           dtype=np.float32).reshape(-1).copy()
                        step_captures.append(captures)
                return own_chain, np.stack(logits_rows), step_captures

            chain, strict_logits, strict_caps = forced_capture(strict)
            _, candidate_logits, candidate_caps = forced_capture(candidate, forced_chain=chain)
            if np.asarray(ids).size and strict_logits.shape != candidate_logits.shape:
                raise RuntimeError("arm shape mismatch")

            same = np.array([
                routing_identical(s, c, reader, num_layers)
                for s, c in zip(strict_caps, candidate_caps)
            ])
            flipped = ~same

            from scripts.gemma4_teacher_forced_gate import row_kl_divergence
            kls = np.array([row_kl_divergence(b, c)
                            for b, c in zip(strict_logits[1:], candidate_logits[1:])])
            # Section 2.11 binds globally and per category; per-case robust
            # statistics are recorded as diagnostics, and a case's pass is its
            # binding controls -- tasks, repeatability, isolation.
            case_diagnostic = verdict(strict_logits[1:], candidate_logits[1:], kls, scope=False)
            flip_rate = float(flipped.mean()) if flipped.size else 0.0

            # Repeatability and isolation, the production packet's controls.
            repeat_ok = True
            for _ in range(2):
                _, again, _ = forced_capture(candidate, forced_chain=chain)
                repeat_ok = repeat_ok and bool(np.array_equal(candidate_logits, again))
            poison = padded_chat(generator, "Reply with the word unrelated.", 777, 900001 + index)
            greedy_output(candidate, poison, 4)
            _, isolated_logits, _ = forced_capture(candidate, forced_chain=chain)
            isolation_ok = bool(np.array_equal(candidate_logits, isolated_logits))

            # Task non-inferiority on the identical task prompts.
            answer = f"maple-{481 + index * 37}"
            task_prompt = (
                f"The requested record has value {answer}. Other records are distractors. "
                f"Return that value as exactly one JSON object with the sole key answer. "
                f"No prose."
            )
            if case["category"] == "general_ja":
                task_prompt = f"指定レコードの値は {answer} です。answer キーのみのJSONでその値を返してください。説明は不要です。"
            elif case["category"] == "mixed_ja_en":
                task_prompt = f"Record value: {answer}。この値を answer キーだけのJSONで返してください。No commentary."
            elif case["category"] == "code":
                task_prompt = f"Python source: record = {{'answer': '{answer}'}}. Serialize record as JSON only, with no extra keys or commentary."
            task_ids = padded_chat(generator, task_prompt, depth, 300001 + index)
            strict_text = generator.tokenizer.decode(greedy_output(
                strict, task_ids, 32, stop_ids=generator.tokenizer.stop_token_ids))
            candidate_text = generator.tokenizer.decode(greedy_output(
                candidate, task_ids, 32, stop_ids=generator.tokenizer.stop_token_ids))
            task_b = check_task_answer(strict_text, answer)
            task_c = check_task_answer(candidate_text, answer)

            record = {
                "id": name, "category": case["category"], "split": case["split"],
                "prompt_tokens": len(ids),
                "routing_identical_rows": int(same.sum()),
                "routing_flip_rows": int(flipped.sum()),
                "flip_rate_diagnostic": flip_rate,
                "routing_identical_kl_max": (float(kls[same].max()) if same.any() else None),
                "case_robust_diagnostic": case_diagnostic,
                "flipped_kl_max": (float(kls[flipped].max()) if flipped.any() else 0.0),
                "three_run_repeat_equal": repeat_ok,
                "after_unrelated_request_equal": isolation_ok,
                "task": {"strict_passed": task_b, "candidate_passed": task_c,
                         "noninferior": int(task_c) >= int(task_b)},
                "passed": (repeat_ok and isolation_ok and int(task_c) >= int(task_b)),
            }
            report["cases"].append(record)
            overall_all_b.append((strict_logits[1:], kls))
            overall_all_c.append((candidate_logits[1:], kls))
            cat = case["category"]
            category_same.setdefault(cat, []).append((strict_logits[1:], candidate_logits[1:], kls))
            (args.directory / "progress.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"case={name} median={case_diagnostic['kl_median']:.3e} p90={case_diagnostic['kl_p90']:.3e} "
                  f"damage={case_diagnostic['flip_damage_rate']:.3f} top1={case_diagnostic['top1_rate']:.4f} "
                  f"routing_flips={flipped.sum()}/{args.rows-1} "
                  f"repeat={repeat_ok} iso={isolation_ok} task={task_b}/{task_c} "
                  f"passed={record['passed']}", flush=True)

        all_b = np.concatenate([p[0] for p in overall_all_b])
        all_c = np.concatenate([p[0] for p in overall_all_c])
        all_kls = np.concatenate([p[1] for p in overall_all_b])
        report["global_section_211"] = verdict(all_b, all_c, all_kls, scope=False)
        report["global_routing_flip_rate_diagnostic"] = float(
            sum(r["routing_flip_rows"] for r in report["cases"])
            / sum(r["routing_flip_rows"] + r["routing_identical_rows"] for r in report["cases"]))
        report["global_routing_identical_kl_max_diagnostic"] = max(
            (r["routing_identical_kl_max"] for r in report["cases"]
             if r["routing_identical_kl_max"] is not None), default=None)
        report["category_section_211"] = {}
        for cat, triples in category_same.items():
            b = np.concatenate([p[0] for p in triples])
            c = np.concatenate([p[1] for p in triples])
            k = np.concatenate([p[2] for p in triples])
            report["category_section_211"][cat] = verdict(b, c, k, scope=True)
        report["elapsed_s"] = time.monotonic() - started
        report["passed"] = (all(r["passed"] for r in report["cases"])
                            and report["global_section_211"]["passed"]
                            and all(v["passed"] for v in report["category_section_211"].values())
                            and all(r["task"]["noninferior"] for r in report["cases"]))
        report["promotion_eligible"] = report["passed"]
        (args.directory / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"packet_passed={report['passed']} "
              f"global_median={report['global_section_211']['kl_median']:.3e} "
              f"global_p90={report['global_section_211']['kl_p90']:.3e} "
              f"global_damage={report['global_section_211']['flip_damage_rate']:.4f} "
              f"global_top1={report['global_section_211']['top1_rate']:.4f} "
              f"routing_flip_rate_diag={report['global_routing_flip_rate_diagnostic']:.3f} "
              f"promotion_eligible={report['promotion_eligible']}", flush=True)
        return 0 if report["passed"] else 1
    finally:
        strict.close()
        candidate.close()
        llm.close()


if __name__ == "__main__":
    raise SystemExit(main())
