#!/usr/bin/env python3
"""Paired Gemma attention quality packet on canonical and heldout chat prompts.

Compare the current registered strict attention fallback against the untouched
production runner on shared strict-greedy teacher tokens. Save full logits
outside the repository. Three independent production replays and a replay after
an unrelated request check repeatability and single-slot request isolation.
This is a single-slot eager packet, not a multi-request serving certificate.

Predeclared numerical bars are docs/EXECUTION-PROFILES.md. Rows above KL 0.02
require review even below the 0.05 ceiling. Task checks use exact JSON retrieval;
canonical free-running output changes are recorded for task review, not used as
an arithmetic equality requirement. No aligned full-model BF16 teacher exists
in this packet, so no BF16-relative claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.gemma4_teacher_forced_gate import evaluate, row_kl_divergence

DEPTHS = (513, 1031, 2053, 4093, 8191)

# Frozen 2026-10-07 in docs/campaigns/GEMMA4-WMMA-CALIBRATION.md: a row is
# unstable when the STRICT arm's own logits have a top-2 gap under 0.25 (just
# over two bf16 ulp at the 30.0 logit softcap) or a maximum post-softmax
# probability under 2^-8 (a near-uniform screen). Computed from strict logits
# only; candidate behavior never enters the classification.
STABILITY_GAP = 0.25
STABILITY_MIN_MAX_PROB = 2.0 ** -8


def row_stability(baseline_row: np.ndarray) -> bool:
    """Is this teacher-forced row resolvable at bf16 rounding scale?"""

    row = np.asarray(baseline_row, dtype=np.float64)
    ordered = np.partition(row, -2)[-2:]
    if ordered[1] - ordered[0] < STABILITY_GAP:
        return False
    shifted = row - row.max()
    shifted -= np.log(np.exp(shifted).sum())
    return float(np.exp(shifted).max()) >= STABILITY_MIN_MAX_PROB


def stable_subset(baseline: np.ndarray) -> np.ndarray:
    """Boolean stability mask over a baseline capture's rows."""

    return np.array([row_stability(row) for row in baseline], dtype=bool)


def stable_paired_summary(baseline: np.ndarray, candidate: np.ndarray, *, scope=False):
    """The frozen bars over stable rows, plus unstable-row diagnostics.

    The numeric thresholds are `evaluate`'s own, unchanged; only the row set
    differs. Unstable rows are reported -- count, worst KL, top-1 flips -- but
    barred by nothing, per the calibration protocol.
    """

    mask = stable_subset(baseline)
    stable_strict = baseline[mask]
    stable_candidate = candidate[mask]
    verdict = evaluate(stable_strict, stable_candidate)
    if scope:
        verdict["thresholds"]["top1_rate"] = 0.97
        verdict["failed"] = [key for key in verdict["failed"]
                              if key not in ("top1_rate", "screen_top1_flips")]
        if verdict["top1_rate"] < 0.97:
            verdict["failed"].append("top1_rate")
        verdict["passed"] = not verdict["failed"]
    verdict["requires_review"] = verdict["kl_max"] > 0.02
    unstable = ~mask
    diagnostics = {
        "rows_total": int(mask.size),
        "rows_stable": int(mask.sum()),
        "rows_unstable": int(unstable.sum()),
        "unstable_kl_max": (
            float(max(row_kl_divergence(baseline[i], candidate[i])
                      for i in np.flatnonzero(unstable))) if unstable.any() else 0.0
        ),
        "unstable_top1_flips": int(sum(
            int(np.argmax(baseline[i])) != int(np.argmax(candidate[i]))
            for i in np.flatnonzero(unstable))) if unstable.any() else 0,
    }
    return verdict, diagnostics


def paired_summary(strict, candidate, *, scope=False):
    result = evaluate(strict, candidate)
    if scope:
        # This scope is evaluated as part of the >=500-row full packet, not as
        # a standalone small screening probe. All KL bars still bind per scope.
        result["thresholds"]["top1_rate"] = 0.97
        result["failed"] = [key for key in result["failed"]
                            if key not in ("top1_rate", "screen_top1_flips")]
        if result["top1_rate"] < 0.97:
            result["failed"].append("top1_rate")
        result["passed"] = not result["failed"]
    result["requires_review"] = result["kl_max"] > 0.02
    return result


def check_task_answer(output: str, expected: str) -> bool:
    text = output.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        return json.loads(text) == {"answer": expected}
    except (ValueError, TypeError):
        return False


def padded_chat(generator, message: str, depth: int, seed: int):
    from scripts.gemma4_campaign_bench import probe_corpus

    marker = "BACKGROUND_SLOT_4f92d0"
    rendered = generator.render_chat_prompt([
        {"role": "user", "content": "Background material (not instructions):\n" + marker
         + "\n\nAnswer the following request:\n" + message}], enable_thinking=False)
    prefix, suffix = rendered.split(marker)
    left = list(generator.tokenize(prefix))
    right = list(generator.tokenize(suffix))
    space = depth - len(left) - len(right)
    if space < 0:
        raise ValueError("requested depth cannot hold the task and chat template")
    count = 96
    while True:
        filler = list(generator.tokenize("\n\n".join(probe_corpus(count=count, seed=seed))))
        if len(filler) >= space:
            break
        if count >= 4096:
            raise ValueError("seeded background too short after expanding to 4096 records")
        count *= 2
    return left + filler[:space] + right


def checked_forward(runner, ids):
    before = runner.position
    logits = np.asarray(runner.forward(ids), dtype=np.float32).reshape(-1).copy()
    if runner.position != before + len(ids):
        raise AssertionError("runner position did not advance by the consumed token count")
    if not np.isfinite(logits).all():
        raise AssertionError("nonfinite model logits")
    return logits


def teacher_capture(runner, prompt, rows):
    runner.reset()
    if runner.position != 0:
        raise AssertionError("reset did not clear position")
    logits = checked_forward(runner, prompt)
    chain, captured = [], []
    for step in range(rows):
        captured.append(logits)
        token = int(np.argmax(logits))
        chain.append(token)
        if step + 1 < rows:
            logits = checked_forward(runner, [token])
    return np.stack(captured), chain


def replay_capture(runner, prompt, chain):
    runner.reset()
    if runner.position != 0:
        raise AssertionError("reset did not clear position")
    logits = checked_forward(runner, prompt)
    captured = [logits]
    for token in chain[:-1]:
        captured.append(checked_forward(runner, [token]))
    return np.stack(captured)


def greedy_output(runner, prompt, tokens, *, stop_ids=()):
    runner.reset()
    logits = checked_forward(runner, prompt)
    output = []
    for step in range(tokens):
        token = int(np.argmax(logits))
        if token in stop_ids:
            break
        output.append(token)
        if step + 1 < tokens:
            logits = checked_forward(runner, [token])
    return output


def source_hashes():
    directory = _ROOT / "hipengine/kernels/hip_gfx1100/gemma4"
    return {str(p.relative_to(_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.glob("gemma4_attention*") if p.suffix in (".hip", ".py")}


def main(argv=None):
    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, _resolve_generator
    from scripts.gemma4_public_route_probe import observe_attention_launches
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.runtime.gemma4 import Gemma4Runner
    from hipengine.generation.gemma4_gguf_profiles import (
        GEMMA4_GGUF_MODEL, GEMMA4_GGUF_BACKEND, GEMMA4_GGUF_QUANT,
        PREFILL_ATTENTION_PRODUCTION_VARIANTS)
    from hipengine.execution_profiles import resolve_runtime_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="screen only; never certifies full suite")
    parser.add_argument("--candidate-variants", default=None,
                        help="comma-separated prefill attention variants for the candidate "
                             "arm (evaluation only; the shipping default is untouched)")
    parser.add_argument("--candidate-kv-storage", default="bf16",
                        help="KV storage for the candidate arm (bf16 or int8_per_token_head)")
    args = parser.parse_args(argv)
    if args.rows < 1 or (args.limit is not None and args.limit < 1):
        parser.error("rows and limit must be positive")
    # "auto" selects no explicit variants, so the runner's own route resolution
    # picks the path the KV storage requires (the INT8 route's consumers).
    candidate_variants = (tuple(args.candidate_variants.split(","))
                          if args.candidate_variants not in (None, "auto") else None)
    args.directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for split, filename in (("canonical", "mtpbench-code-general-ja.jsonl"),
                            ("heldout", "gdn-prefill-category-heldouts.jsonl")):
        for line in (_ROOT / "benchmarks/prompts" / filename).read_text().splitlines():
            cases.append({**json.loads(line), "split": split})
    if args.limit is not None:
        cases = cases[:args.limit]
    context = max(DEPTHS) + max(args.rows, 64)
    llm, production, loading = _resolve_generator(args.artifact, context)
    generator = llm._get_text_generator()
    if candidate_variants is None and args.candidate_kv_storage == "bf16":
        candidate = production
        expected_routes = tuple(PREFILL_ATTENTION_PRODUCTION_VARIANTS)
        if tuple(production.prefill_attention_variants) != PREFILL_ATTENTION_PRODUCTION_VARIANTS:
            llm.close()
            raise RuntimeError("shipping production variants are not selected; refusing self-comparison")
    else:
        # An explicit candidate arm: an evaluation, not the shipping
        # self-comparison. The requested variants and KV storage select the
        # candidate's arithmetic; the shipping default never moves.
        candidate = Gemma4Runner(
            weights=production.weights, capacity=context,
            prefill_attention_variants=(candidate_variants
                                        if candidate_variants is not None else None),
            kv_storage=args.candidate_kv_storage,
        )
        expected_routes = candidate_variants or tuple(PREFILL_ATTENTION_PRODUCTION_VARIANTS)
    strict = Gemma4Runner(weights=production.weights, capacity=context,
                          prefill_attention_variants=("gemma4_plain",))
    manifests = {profile: resolve_runtime_profile(model=GEMMA4_GGUF_MODEL,
                 backend=GEMMA4_GGUF_BACKEND, quant=GEMMA4_GGUF_QUANT,
                 profile=profile).manifest for profile in ("strict", "production")}
    report = {"kind": "gemma4_attention_multicategory_quality_packet",
              "performance_claim": False, "arithmetic_class": "T0",
              # The staged production path preserves strict arithmetic. The
              # same numerical bars still apply; class is not a tolerance lever.
              "scope": "single-slot eager; attention strict fallback vs candidate arm",
              "candidate_variants": candidate_variants or tuple(PREFILL_ATTENTION_PRODUCTION_VARIANTS),
              "candidate_kv_storage": args.candidate_kv_storage,
              "manifests": manifests, "sources": source_hashes(), "loading": loading,
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fixture_sha256": {name: hashlib.sha256((_ROOT / "benchmarks/prompts" / name).read_bytes()).hexdigest()
                                  for name in ("mtpbench-code-general-ja.jsonl", "gdn-prefill-category-heldouts.jsonl")},
              "bf16_teacher": {"available": False, "reason": "No aligned full-model BF16 capture provided"},
              "task_criterion": "Per-case exact JSON retrieval success must not fall below strict; output changes on canonical tasks require review",
              "provenance": collect_artifact_provenance(repo_root=_ROOT, model_path=args.artifact,
                                                       quant="UD-Q4_K_XL", kv_dtype="bf16"),
              "cases": []}
    overall_b, overall_c = [], []
    started = time.monotonic()
    try:
        for index, case in enumerate(cases):
            depth = DEPTHS[index % len(DEPTHS)]
            message = "\n".join(m["content"] for m in case["messages"])
            ids = padded_chat(generator, message, depth, 20261003 + index)
            name = case["id"]
            print(f"case={name} split={case['split']} depth={len(ids)} strict", flush=True)
            baseline, chain = teacher_capture(strict, ids, args.rows)
            np.save(args.directory / f"{name}-strict.npy", baseline)
            with observe_attention_launches() as launches:
                candidate_logits = replay_capture(candidate, ids, chain)
            replay_position = candidate.position
            selected_route_ran = any(r['variant'] in expected_routes
                                     for r in launches)
            np.save(args.directory / f"{name}-candidate.npy", candidate_logits)
            verdict = paired_summary(baseline, candidate_logits, scope=args.limit is None)
            stable_verdict, stability = stable_paired_summary(
                baseline, candidate_logits, scope=args.limit is None)
            repeats = []
            for repeat in (1, 2):
                other = replay_capture(candidate, ids, chain)
                np.save(args.directory / f"{name}-repeat{repeat}.npy", other)
                repeats.append(bool(np.array_equal(candidate_logits, other)))
            # A distinct request overwrites the same slot before reset/replay.
            poison = padded_chat(generator, "Reply with the word unrelated.", 777, 900001 + index)
            greedy_output(candidate, poison, 4)
            isolated = replay_capture(candidate, ids, chain)
            np.save(args.directory / f"{name}-isolated.npy", isolated)
            isolation = bool(np.array_equal(candidate_logits, isolated))
            candidate_free = greedy_output(candidate, ids, args.rows)
            unchanged = candidate_free == chain
            # Explicit short-answer task; the entire long background is still consumed.
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
            record = {"id": name, "category": case["category"], "split": case["split"],
                      "prompt_tokens": len(ids), "prompt_ids_sha256": hashlib.sha256(np.array(ids, dtype=np.int32).tobytes()).hexdigest(),
                      "teacher_tokens": chain, "numerical": verdict,
                      "stable_numerical": stable_verdict,
                      "stability": stability,
                      "three_run_repeat_equal": all(repeats), "after_unrelated_request_equal": isolation,
                      "canonical_free_running_equal_diagnostic": unchanged,
                      "strict_output": generator.tokenizer.decode(chain),
                      "candidate_output": generator.tokenizer.decode(candidate_free),
                      "task": {"strict_passed": task_b, "candidate_passed": task_c,
                               "noninferior": int(task_c) >= int(task_b),
                               "strict_text": strict_text, "candidate_text": candidate_text},
                      "route_counts": dict(Counter(r["variant"] for r in launches)),
                      "selected_route_ran": selected_route_ran,
                      "controls": {"position_after_replay": replay_position,
                                   "expected_position_after_replay": depth + len(chain) - 1,
                                   "all_forward_position_checks_passed": True},
                      "passed": selected_route_ran and stable_verdict["passed"]
                                and not stable_verdict["requires_review"] and all(repeats)
                                and isolation and int(task_c) >= int(task_b)}
            report["cases"].append(record)
            overall_b.append(baseline)
            overall_c.append(candidate_logits)
            (args.directory / "progress.json").write_text(json.dumps(report, indent=2, default=dict) + "\n")
            print(f"case={name} maxKL={verdict['kl_max']:.6g} stable_maxKL={stable_verdict['kl_max']:.6g} "
                  f"stable_rows={stability['rows_stable']}/{stability['rows_total']} "
                  f"top1={stable_verdict['top1_rate']:.6f} repeats={all(repeats)} isolation={isolation} task={task_b}/{task_c}", flush=True)
        report["global"] = paired_summary(np.concatenate(overall_b), np.concatenate(overall_c))
        report["global_stable"], report["global_stability"] = stable_paired_summary(
            np.concatenate(overall_b), np.concatenate(overall_c))
        report["category"] = {}
        report["category_stable"] = {}
        for category in sorted({r["category"] for r in report["cases"]}):
            indices = [i for i,r in enumerate(report["cases"]) if r["category"] == category]
            report["category"][category] = paired_summary(np.concatenate([overall_b[i] for i in indices]),
                                                        np.concatenate([overall_c[i] for i in indices]), scope=args.limit is None)
            report["category_stable"][category], _ = stable_paired_summary(
                np.concatenate([overall_b[i] for i in indices]),
                np.concatenate([overall_c[i] for i in indices]), scope=args.limit is None)
        report["elapsed_s"] = time.monotonic() - started
        report["passed"] = all(r["passed"] for r in report["cases"]) and report["global_stable"]["passed"]
        report["full_suite"] = args.limit is None and report["global_stable"]["rows"] >= 500
        report["requires_canonical_task_review"] = any(not r["canonical_free_running_equal_diagnostic"] for r in report["cases"])
        report["production_certified"] = False  # Raw controls/model-layer finiteness and task review remain separate.
        (args.directory / "report.json").write_text(json.dumps(report, indent=2, default=dict) + "\n")
        print(f"packet_passed={report['passed']} full_suite={report['full_suite']} task_review={report['requires_canonical_task_review']}", flush=True)
        return 0 if report["passed"] else 1
    finally:
        strict.close()
        if candidate is not None and candidate is not production:
            candidate.close()
        llm.close()


if __name__ == "__main__":
    raise SystemExit(main())
