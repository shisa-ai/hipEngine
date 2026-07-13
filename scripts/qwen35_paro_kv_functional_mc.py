#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a bounded long-context multiple-choice PARO BF16-vs-INT8 KV check.

This is the scorable replacement for the free-generation smoke whose BF16
reference answered 0/5. Each natural task is expanded to an exact context,
then both policies consume the same fixed assistant prefix. The expected answer
is scored only among four declared A-D option tokens. A task is candidate-
scorable only when BF16 first selects its known-correct option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kvcache import ResolvedKVPolicy, resolve_kv_policy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy
from scripts.qwen35_paro_int8_kv_quality_sweep import _kv_memory_audit, _read_logits
from scripts.qwen35_paro_kv_quality_smoke import (
    CATEGORIES,
    SuiteError,
    _build_prompt_tokens,
    _load_suite,
    _read_compiler_version,
    _select_tasks,
)

DEFAULT_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/"
    "models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/"
    "snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1"
)
DEFAULT_SUITE = REPO_ROOT / "benchmarks" / "prompts" / "kv-int8-long-context-mc.jsonl"
ANSWER_PREFIX = "The correct option is "
CHOICES = ("A", "B", "C", "D")


def _validate_mc_tasks(tasks: Sequence[dict[str, Any]]) -> None:
    for task in tasks:
        choices = task.get("choices")
        if not isinstance(choices, dict) or tuple(choices) != CHOICES:
            raise SuiteError(f"task {task['id']!r} choices must be ordered A, B, C, D")
        if any(not isinstance(choices[label], str) or not choices[label] for label in CHOICES):
            raise SuiteError(f"task {task['id']!r} choices must contain non-empty answer text")
        expected = task.get("expected_choice")
        if expected not in CHOICES:
            raise SuiteError(f"task {task['id']!r} expected_choice must be A, B, C, or D")
        if str(choices[str(expected)]) not in [str(item) for item in task["expected"]]:
            raise SuiteError(f"task {task['id']!r} expected choice text must match expected answers")


def _single_token_ids(tokenizer: Any, texts: Sequence[str], *, label: str) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        encoded = [int(token) for token in tokenizer.encode(text).ids]
        if len(encoded) != 1:
            raise SuiteError(f"{label} {text!r} must encode to one token, got {encoded}")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise SuiteError(f"{label} token IDs must be distinct")
    return token_ids


def score_choices(
    logits: np.ndarray,
    *,
    choice_token_ids: Sequence[int],
    expected_choice: str,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size <= max(int(token) for token in choice_token_ids):
        raise ValueError("choice token lies outside logits")
    selected_values = np.asarray([values[int(token)] for token in choice_token_ids], dtype=np.float64)
    shifted = selected_values - float(np.max(selected_values))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities))
    selected_index = int(np.argmax(selected_values))
    expected_index = CHOICES.index(expected_choice)
    strongest_wrong = float(np.max(np.delete(selected_values, expected_index)))
    return {
        "selected_choice": CHOICES[selected_index],
        "expected_choice": expected_choice,
        "passed": selected_index == expected_index,
        "choice_logits": {choice: float(selected_values[index]) for index, choice in enumerate(CHOICES)},
        "restricted_probabilities": {
            choice: float(probabilities[index]) for index, choice in enumerate(CHOICES)
        },
        "expected_margin_vs_strongest_wrong": float(selected_values[expected_index] - strongest_wrong),
        "full_vocab_top1": int(np.argmax(values)),
        "finite_logits": bool(np.all(np.isfinite(values))),
        "logits_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _run_policy(
    *,
    runner: Qwen35ParoNextTokenRunner,
    policy: ResolvedKVPolicy,
    tasks: Sequence[dict[str, Any]],
    expanded_prompts: dict[str, tuple[list[int], dict[str, Any]]],
    answer_prefix_token_ids: Sequence[int],
    choice_token_ids: Sequence[int],
    max_sequence_length: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reset_memory_stats()
    rows: list[dict[str, Any]] = []
    logits_by_id: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512, auto_tune_chunk_sizes=True),
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        for task in tasks:
            task_id = str(task["id"])
            prompt_tokens, prompt_metadata = expanded_prompts[task_id]
            session.reset()
            session._resolve_prefill_config_for_length(len(prompt_tokens))
            task_start = time.perf_counter()
            session.prefill_native(prompt_tokens, sample=False)
            prefix_top1: list[int] = []
            for offset, token_id in enumerate(answer_prefix_token_ids):
                result = session.step(
                    int(token_id),
                    position=len(prompt_tokens) + offset,
                    sample=True,
                )
                if result is None:
                    raise RuntimeError(f"task {task_id!r} answer-prefix step did not sample")
                prefix_top1.append(int(result.token_id))
            logits = _read_logits(session)
            logits_by_id[task_id] = logits
            rows.append(
                {
                    "id": task_id,
                    "category": str(task["category"]),
                    **prompt_metadata,
                    "choices": {choice: str(task["choices"][choice]) for choice in CHOICES},
                    "answer_prefix_top1": prefix_top1,
                    "elapsed_seconds": time.perf_counter() - task_start,
                    "score": score_choices(
                        logits,
                        choice_token_ids=choice_token_ids,
                        expected_choice=str(task["expected_choice"]),
                    ),
                }
            )
        owned_summary = session.owned_buffer_summary()
    elapsed = time.perf_counter() - started
    memory_audit = _kv_memory_audit(owned_summary, policy.storage_dtype.value)
    return (
        {
            "kv_policy": kv_policy_json(policy),
            "rows": rows,
            "summary": {
                "passed": sum(bool(row["score"]["passed"]) for row in rows),
                "total": len(rows),
                "score": float(sum(bool(row["score"]["passed"]) for row in rows) / len(rows)),
                "elapsed_seconds": elapsed,
            },
            "memory": memory_stats(),
            "memory_audit": memory_audit,
            "owned_buffer_summary": {
                key: owned_summary.get(key)
                for key in (
                    "allocation_bytes",
                    "buffer_bytes",
                    "full_attention_kv_payload_bytes",
                    "full_attention_kv_scale_bytes",
                    "full_attention_kv_total_bytes",
                    "kv_storage_dtype",
                    "kv_scale_dtype",
                    "kv_scale_granularity",
                )
            },
        },
        logits_by_id,
    )


def pair_policy_results(
    tasks: Sequence[dict[str, Any]],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_logits: dict[str, np.ndarray],
    candidate_logits: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    reference_rows = {str(row["id"]): row for row in reference["rows"]}
    candidate_rows = {str(row["id"]): row for row in candidate["rows"]}
    paired: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        ref_score = reference_rows[task_id]["score"]
        cand_score = candidate_rows[task_id]["score"]
        correctness = evaluate_logits(reference_logits[task_id], candidate_logits[task_id])
        qualified = bool(ref_score["passed"])
        retained = bool(qualified and cand_score["passed"])
        paired.append(
            {
                "id": task_id,
                "category": str(task["category"]),
                "reference_qualified": qualified,
                "reference_choice": str(ref_score["selected_choice"]),
                "candidate_choice": str(cand_score["selected_choice"]),
                "expected_choice": str(task["expected_choice"]),
                "candidate_retained": retained,
                "candidate_regression": bool(qualified and not cand_score["passed"]),
                "choice_match": ref_score["selected_choice"] == cand_score["selected_choice"],
                "full_logits": {
                    "mean_kl": correctness.kl_mean,
                    "max_kl": correctness.kl_max,
                    "top1_agreement": correctness.top1_agreement,
                },
            }
        )
    qualified = [row for row in paired if row["reference_qualified"]]
    regressions = [row for row in qualified if row["candidate_regression"]]
    if len(qualified) != len(paired):
        status = "partially_scorable" if qualified else "reference_unscorable"
    elif regressions:
        status = "candidate_quality_regression"
    else:
        status = "accepted_bounded_smoke"
    summary = {
        "reference_qualified": len(qualified),
        "total": len(paired),
        "candidate_retained": sum(bool(row["candidate_retained"]) for row in qualified),
        "candidate_regressions": [str(row["id"]) for row in regressions],
        "reference_failures": [str(row["id"]) for row in paired if not row["reference_qualified"]],
        "fully_scorable": len(qualified) == len(paired),
    }
    return paired, status, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _select_tasks(_load_suite(args.suite), categories=args.categories, limit=args.limit)
    _validate_mc_tasks(tasks)
    candidate_policy = resolve_args_kv_policy(args, block_size=256)
    reference_policy = resolve_kv_policy("bf16", block_size=256)
    compiler_version = _read_compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(
        args.model,
        shared_expert_format=None if args.shared_expert_format == "auto" else args.shared_expert_format,
        backend=args.backend,
    )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    choice_token_ids = _single_token_ids(tokenizer, CHOICES, label="choice")
    answer_prefix_token_ids = [int(token) for token in tokenizer.encode(ANSWER_PREFIX).ids]
    if not answer_prefix_token_ids:
        raise SuiteError("answer prefix must tokenize to at least one token")
    expanded = {
        str(task["id"]): _build_prompt_tokens(tokenizer, task, context_tokens=args.context_tokens or None)
        for task in tasks
    }
    max_context = max(len(tokens) for tokens, _metadata in expanded.values())
    max_sequence_length = max_context + len(answer_prefix_token_ids) + 2
    reference, reference_logits = _run_policy(
        runner=runner,
        policy=reference_policy,
        tasks=tasks,
        expanded_prompts=expanded,
        answer_prefix_token_ids=answer_prefix_token_ids,
        choice_token_ids=choice_token_ids,
        max_sequence_length=max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    candidate, candidate_logits = _run_policy(
        runner=runner,
        policy=candidate_policy,
        tasks=tasks,
        expanded_prompts=expanded,
        answer_prefix_token_ids=answer_prefix_token_ids,
        choice_token_ids=choice_token_ids,
        max_sequence_length=max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    paired, status, summary = pair_policy_results(
        tasks,
        reference,
        candidate,
        reference_logits,
        candidate_logits,
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=args.backend,
        resolved_backend=runner.backend,
        target_arch=runner.target_arch,
        model_path=args.model,
        quant="w4_paro",
        kv_dtype=f"bf16_vs_{candidate_policy.storage_dtype.value}",
        command=(sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]),
        environment={
            key: os.environ.get(key)
            for key in ("HIP_VISIBLE_DEVICES", "HIPENGINE_HIP_ARCH", "HIPENGINE_BACKEND")
        },
        build_profile="kv_int8_functional_mc",
        timing_protocol="serial restricted-choice tasks; no warmup; timing diagnostic only",
        warmups=0,
        repetitions=1,
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": "qwen35_paro_kv_bounded_functional_mc",
        "status": status,
        "performance_claim": False,
        "full_benchmark_claim": False,
        "model": str(args.model),
        "backend": runner.backend,
        "target_arch": runner.target_arch,
        "protocol": {
            "scoring": "restricted A-D next-token choice after fixed assistant answer prefix",
            "answer_prefix": ANSWER_PREFIX,
            "answer_prefix_token_ids": answer_prefix_token_ids,
            "choice_token_ids": dict(zip(CHOICES, choice_token_ids, strict=True)),
            "candidate_scorable_only_if_reference_correct": True,
            "free_generation_claim": False,
        },
        "suite": {
            "path": str(args.suite),
            "sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "selected_ids": [str(task["id"]) for task in tasks],
            "categories": [str(task["category"]) for task in tasks],
            "context_tokens_override": int(args.context_tokens) if args.context_tokens else None,
        },
        "provenance": provenance,
        "reference": reference,
        "candidate": candidate,
        "paired": paired,
        "summary": summary,
        "notes": [
            "This five-task restricted-choice probe is bounded functional evidence, not a full long-context benchmark or free-generation quality claim.",
            "The fixed assistant prefix makes the scored A-D token position identical across BF16 and INT8 and ensures INT8 K/V is consumed after prefill.",
            "A BF16 miss is excluded from candidate retention rather than counted as an INT8 pass.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument(
        "--shared-expert-format",
        choices=("auto", "legacy_fp16", "packed_paro_w4"),
        default="packed_paro_w4",
    )
    add_kv_policy_args(
        parser,
        default_storage="int8_per_token_head",
        help_prefix="Candidate KV storage for the bounded functional multiple-choice check",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.context_tokens <= 0:
        parser.error("--context-tokens must be positive")
    if args.max_layers < 0:
        parser.error("--max-layers must be non-negative")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
