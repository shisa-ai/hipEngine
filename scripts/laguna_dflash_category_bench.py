#!/usr/bin/env python3
"""Benchmark Laguna DFlash B4 against true AR on the canonical category suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Iterable

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.benchmark.speculative import (
    SpeculativeBenchmarkModels,
    build_speculative_artifact,
)
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.speculative.laguna_dflash import (
    LagunaDFlashResidentCycle,
    LagunaDFlashResidentDrafter,
)
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_ORACLE,
    DEFAULT_ORACLE_LOGPROBS,
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    EXPECTED_CATEGORIES,
    EXPECTED_PROMPT_COUNT,
    _compiler_version,
    _load_prompts,
    _oracle_gate,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFTER = Path(
    "/home/lhl/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash/"
    "snapshots/b0486d1586daa0d56435c508108171fc1c8daff9"
)
DEFAULT_DRAFTER_REVISION = "b0486d1586daa0d56435c508108171fc1c8daff9"
DEFAULT_DRAFTER_SHA256 = "f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4"
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-dflash-category-economics-post-prefill.json"
)
DEFAULT_HELDOUT_IDS = frozenset(
    (
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    )
)
ADMITTED_BUDGET = 4
DEFAULT_OUTPUT_TOKENS = 32
DEFAULT_REPETITIONS = 2
RETAINED_CHUNK_SIZE = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("drafter", nargs="?", type=Path, default=DEFAULT_DRAFTER)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--oracle-logprobs", type=Path, default=DEFAULT_ORACLE_LOGPROBS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=RETAINED_CHUNK_SIZE)
    parser.add_argument("--candidate-budget", type=int, default=ADMITTED_BUDGET)
    parser.add_argument(
        "--iq3-selected-down-tile",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="explicit gfx1100 IQ3 selected-down output tile (default: baseline tile1)",
    )
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-output-tokens", type=int, default=6)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-gguf", action="store_true")
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--quant-label", default="Q4_K_M mixed GGUF v3 / hipEngine repacked-v1")
    parser.add_argument("--drafter-sha256", default=DEFAULT_DRAFTER_SHA256)
    parser.add_argument("--drafter-revision", default=DEFAULT_DRAFTER_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _fixed_horizon_state_aligned(
    *,
    target_position: int,
    drafter_context_tokens: int,
    expected_prediction_position: int,
) -> bool:
    """Accept either valid fixed-horizon commit boundary.

    A correction/bonus remains predicted but uncommitted at ``expected``. When
    the remaining-output limit suppresses that bonus, its accepted final row is
    already committed at ``expected + 1``. Both own the same exact visible
    output, and the drafter must align to the target in either case.
    """

    return bool(
        int(target_position)
        in (int(expected_prediction_position), int(expected_prediction_position) + 1)
        and int(drafter_context_tokens) == int(target_position) + 1
    )


def _resolved_target_prefill_variants(
    target: LagunaGGUFResidentSession,
) -> tuple[str, str]:
    if target.kv_cache is None:
        raise RuntimeError("Laguna target KV cache is unavailable")
    global_variant = next(
        state.attention_prefill_variant
        for state in target.kv_cache.layers
        if state.attention_type == FULL_ATTENTION
    )
    return str(global_variant), str(target.swa_prefill_variant)


def _reset_request(
    target: LagunaGGUFResidentSession,
    drafter: LagunaDFlashResidentDrafter,
) -> None:
    target.reset_state()
    drafter.reset_state()
    if target.position != -1 or drafter.committed_context_tokens != 0:
        raise RuntimeError("Laguna target/drafter request reset did not return to position zero")


def _run_ar(
    target: LagunaGGUFResidentSession,
    prompt: dict[str, Any],
    *,
    output_tokens: int,
) -> dict[str, Any]:
    prefill_started = time.perf_counter()
    result = target.prefill(prompt["token_ids"], use_bulk=True)
    target.runtime.device_synchronize()
    ttft_seconds = time.perf_counter() - prefill_started
    generated = [int(result.next_token_id)]
    logits_finite = math.isfinite(float(result.next_token_logit))
    decode_started = time.perf_counter()
    while len(generated) < output_tokens:
        result = target.forward_token(result.next_token_id)
        generated.append(int(result.next_token_id))
        logits_finite = logits_finite and math.isfinite(float(result.next_token_logit))
    target.runtime.device_synchronize()
    decode_seconds = time.perf_counter() - decode_started
    expected_position = len(prompt["token_ids"]) + output_tokens - 2
    return {
        "mode": "ar",
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_tokens": prompt["prompt_tokens"],
        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
        "generated_ids": generated,
        "generated_ids_sha256": _sha256_json(generated),
        "output_tokens": int(output_tokens),
        "decode_output_tokens": int(output_tokens - 1),
        "ttft_seconds": ttft_seconds,
        "prefill_tok_s": prompt["prompt_tokens"] / ttft_seconds,
        "decode_seconds": decode_seconds,
        "decode_tok_s": (output_tokens - 1) / decode_seconds,
        "total_seconds": ttft_seconds + decode_seconds,
        "e2e_output_tok_s": output_tokens / (ttft_seconds + decode_seconds),
        "finite_logits": bool(logits_finite),
        "target_position": target.position,
        "expected_target_position": expected_position,
        "state_aligned": target.position == expected_position,
    }


def _run_dflash(
    target: LagunaGGUFResidentSession,
    drafter: LagunaDFlashResidentDrafter,
    cycle: LagunaDFlashResidentCycle,
    prompt: dict[str, Any],
    *,
    output_tokens: int,
) -> dict[str, Any]:
    prefill_started = time.perf_counter()
    prompt_result = cycle.prefill(prompt["token_ids"])
    target.runtime.device_synchronize()
    ttft_seconds = time.perf_counter() - prefill_started
    generated = [int(prompt_result.next_token_id)]
    root = generated[0]
    finite_verify = math.isfinite(float(prompt_result.next_token_logit))
    finite_draft = True
    cycle_rows: list[dict[str, Any]] = []
    decode_started = time.perf_counter()
    while len(generated) < output_tokens:
        remaining = output_tokens - len(generated)
        cycle_started = time.perf_counter()
        result = cycle.run_cycle(root, remaining_decode=remaining)
        target.runtime.device_synchronize()
        cycle_wall_seconds = time.perf_counter() - cycle_started
        visible = tuple(int(value) for value in result.visible_output_ids)
        if not visible or len(visible) > remaining:
            raise RuntimeError("Laguna DFlash cycle emitted an invalid visible-output count")
        generated.extend(visible)
        finite_draft = finite_draft and _finite(
            value for row in result.proposal.topk_values for value in row
        )
        finite_verify = finite_verify and _finite(result.target_result.target_top1_values)
        residual_seconds = max(
            0.0,
            cycle_wall_seconds - result.proposal_seconds - result.target_verify_seconds,
        )
        cycle_rows.append(
            {
                "candidate_token_ids": list(result.proposal.candidate_token_ids),
                "target_top1_ids": list(result.target_result.target_top1_ids),
                "accepted_draft_tokens": result.target_result.accepted_draft_count,
                "visible_output_tokens": len(visible),
                "target_verify_rows": len(result.target_batch.tokens),
                "committed_rows": result.target_result.committed_rows,
                "rejected_target_rows": (
                    len(result.target_batch.tokens) - result.target_result.committed_rows
                ),
                "full_accept": result.target_result.full_accept,
                "proposal_seconds": result.proposal_seconds,
                "target_verify_seconds": result.target_verify_seconds,
                "draft_commit_enqueue_seconds": result.draft_commit_enqueue_seconds,
                "cycle_host_seconds": result.cycle_host_seconds,
                "cycle_wall_seconds": cycle_wall_seconds,
                "post_verify_residual_seconds": residual_seconds,
                "verifier_addresses_stable": result.verifier_addresses_stable,
            }
        )
        next_token = result.target_result.next_token_id
        if next_token is None:
            if len(generated) != output_tokens:
                raise RuntimeError("Laguna DFlash stopped before the fixed output horizon")
            break
        root = int(next_token)
    target.runtime.device_synchronize()
    decode_seconds = time.perf_counter() - decode_started
    accepted = sum(int(row["accepted_draft_tokens"]) for row in cycle_rows)
    target_rows = sum(int(row["target_verify_rows"]) for row in cycle_rows)
    proposed = drafter.candidate_budget * len(cycle_rows)
    expected_position = len(prompt["token_ids"]) + output_tokens - 2
    state_aligned = _fixed_horizon_state_aligned(
        target_position=target.position,
        drafter_context_tokens=drafter.committed_context_tokens,
        expected_prediction_position=expected_position,
    )
    return {
        "mode": "dflash",
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_tokens": prompt["prompt_tokens"],
        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
        "generated_ids": generated,
        "generated_ids_sha256": _sha256_json(generated),
        "output_tokens": int(output_tokens),
        "decode_output_tokens": int(output_tokens - 1),
        "ttft_seconds": ttft_seconds,
        "prefill_tok_s": prompt["prompt_tokens"] / ttft_seconds,
        "decode_seconds": decode_seconds,
        "decode_tok_s": (output_tokens - 1) / decode_seconds,
        "total_seconds": ttft_seconds + decode_seconds,
        "e2e_output_tok_s": output_tokens / (ttft_seconds + decode_seconds),
        "finite_draft_logits": bool(finite_draft),
        "finite_verify_logits": bool(finite_verify),
        "cycles": cycle_rows,
        "cycle_count": len(cycle_rows),
        "accepted_lengths": [int(row["accepted_draft_tokens"]) for row in cycle_rows],
        "accepted_draft_tokens": accepted,
        "draft_tokens_proposed": proposed,
        "target_verify_rows": target_rows,
        "target_verify_rows_per_output_token": target_rows / (output_tokens - 1),
        "draft_acceptance": accepted / proposed,
        "accepted_per_output": accepted / (output_tokens - 1),
        "proposal_seconds": sum(float(row["proposal_seconds"]) for row in cycle_rows),
        "target_verify_seconds": sum(float(row["target_verify_seconds"]) for row in cycle_rows),
        "draft_commit_enqueue_seconds": sum(
            float(row["draft_commit_enqueue_seconds"]) for row in cycle_rows
        ),
        "cycle_host_seconds": sum(float(row["cycle_host_seconds"]) for row in cycle_rows),
        "cycle_wall_seconds": sum(float(row["cycle_wall_seconds"]) for row in cycle_rows),
        "post_verify_residual_seconds": sum(
            float(row["post_verify_residual_seconds"]) for row in cycle_rows
        ),
        "all_verifier_addresses_stable": all(
            bool(row["verifier_addresses_stable"]) for row in cycle_rows
        ),
        "target_position": target.position,
        "expected_target_position": expected_position,
        "drafter_context_tokens": drafter.committed_context_tokens,
        "final_output_committed": target.position == expected_position + 1,
        "state_aligned": state_aligned,
    }


def _pair_rows(
    rows: list[dict[str, Any]],
    *,
    candidate_budget: int,
    peak_allocated_bytes: int,
    allocated_after_load_bytes: int,
    backend: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["prompt_id"]), int(row["repetition"]))][str(row["mode"])] = row
    pairs: list[dict[str, Any]] = []
    for (prompt_id, repetition), modes in sorted(grouped.items()):
        if set(modes) != {"ar", "dflash"}:
            raise ValueError(f"missing AR/DFlash pair for {prompt_id} rep {repetition}")
        ar = modes["ar"]
        spec = modes["dflash"]
        if ar["category"] != spec["category"]:
            raise ValueError(f"category mismatch for {prompt_id} rep {repetition}")
        exact = ar["generated_ids"] == spec["generated_ids"]
        pairs.append(
            {
                "prompt": {
                    "id": prompt_id,
                    "dataset": "mtpbench-code-general-ja",
                    "category": ar["category"],
                    "prompt_tokens": ar["prompt_tokens"],
                    "prompt_ids_sha256": ar["prompt_token_ids_sha256"],
                    "representative": True,
                },
                "config": {
                    "name": f"laguna_dflash_chain_b{candidate_budget}",
                    "provider": "dflash",
                    "proposal_mode": "chain",
                    "verify_mode": "verify_chain",
                    "draft_budget": candidate_budget,
                    "topk": 1,
                },
                "ar": {
                    "same_session_control": True,
                    "same_process_control": True,
                    "decode_seconds": ar["decode_seconds"],
                    "finite_logits": ar["finite_logits"],
                    "generated_ids": ar["generated_ids"],
                },
                "spec": {
                    "decode_seconds": spec["decode_seconds"],
                    "draft_seconds": spec["proposal_seconds"],
                    "target_verify_seconds": spec["target_verify_seconds"],
                    "commit_seconds": spec["post_verify_residual_seconds"],
                    "draft_context_append_seconds": spec["draft_commit_enqueue_seconds"],
                    "target_verify_rows": spec["target_verify_rows"],
                    "target_forward_calls": spec["cycle_count"],
                    "target_bulk_forward_calls": spec["cycle_count"],
                    "target_serial_forward_calls": 0,
                    "draft_tokens": spec["draft_tokens_proposed"],
                    "accepted_draft_tokens": spec["accepted_draft_tokens"],
                    "accepted_lengths": spec["accepted_lengths"],
                    "finite_draft_logits": spec["finite_draft_logits"],
                    "finite_verify_logits": spec["finite_verify_logits"],
                    "generated_ids": spec["generated_ids"],
                    "same_session_control": True,
                    "same_process_control": True,
                    "verifier_mode": "native_bulk_b_plus_one",
                    "verifier_state_strategy": "staged_rows_commit_accepted_prefix",
                    "canonical_commit_mode": "accepted_prefix_kv_append",
                    "native_bulk_verifier": True,
                    "backend": backend,
                    "target_arch": backend.removeprefix("hip_"),
                    "d2h": {
                        "scalar_reads": 0,
                        "vector_reads": 5 * spec["cycle_count"],
                        "scalar_values": 0,
                        "vector_values": spec["cycle_count"] * (4 * candidate_budget + 9),
                        "full_logits_readbacks": 0,
                    },
                    "graph": {
                        "status": "not_captured",
                        "replay_steps": 0,
                        "bucket_key": {
                            "mode": "verify_chain",
                            "draft_budget": candidate_budget,
                            "active_c": 1,
                        },
                        "validation_passed": None,
                    },
                },
                "quality_gate": {
                    "exact_match_ar": exact,
                    "finite_ar_logits": ar["finite_logits"],
                    "finite_dflash_draft_logits": spec["finite_draft_logits"],
                    "finite_dflash_verify_logits": spec["finite_verify_logits"],
                },
                "memory": {
                    "allocated_after_load_bytes": allocated_after_load_bytes,
                    "peak_allocated_bytes": peak_allocated_bytes,
                },
                "decode_tokens": ar["decode_output_tokens"],
                "repetition": repetition,
                "split": "heldout" if prompt_id in DEFAULT_HELDOUT_IDS else "train",
                "raw": {"ar": ar, "dflash": spec},
            }
        )
    return pairs


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("benchmark timing denominator must be positive")
    return numerator / denominator


def _aggregate_scope(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("cannot aggregate an empty Laguna DFlash scope")
    ar_decode_seconds = sum(float(row["ar"]["decode_seconds"]) for row in pairs)
    spec_decode_seconds = sum(float(row["spec"]["decode_seconds"]) for row in pairs)
    decode_tokens = sum(int(row["decode_tokens"]) for row in pairs)
    ar_ttft_seconds = sum(float(row["raw"]["ar"]["ttft_seconds"]) for row in pairs)
    spec_ttft_seconds = sum(float(row["raw"]["dflash"]["ttft_seconds"]) for row in pairs)
    output_tokens = sum(int(row["raw"]["ar"]["output_tokens"]) for row in pairs)
    ar_total_seconds = sum(float(row["raw"]["ar"]["total_seconds"]) for row in pairs)
    spec_total_seconds = sum(float(row["raw"]["dflash"]["total_seconds"]) for row in pairs)
    accepted = sum(int(row["raw"]["dflash"]["accepted_draft_tokens"]) for row in pairs)
    proposed = sum(int(row["raw"]["dflash"]["draft_tokens_proposed"]) for row in pairs)
    target_rows = sum(int(row["raw"]["dflash"]["target_verify_rows"]) for row in pairs)
    cycles = sum(int(row["raw"]["dflash"]["cycle_count"]) for row in pairs)
    ar_decode_tok_s = _ratio(decode_tokens, ar_decode_seconds)
    spec_decode_tok_s = _ratio(decode_tokens, spec_decode_seconds)
    return {
        "prompt_runs": len(pairs),
        "unique_prompts": len({str(row["prompt"]["id"]) for row in pairs}),
        "prompt_ids": sorted({str(row["prompt"]["id"]) for row in pairs}),
        "decode_output_tokens": decode_tokens,
        "output_tokens_including_first": output_tokens,
        "ar": {
            "ttft_seconds": ar_ttft_seconds,
            "ttft_median_seconds": statistics.median(
                float(row["raw"]["ar"]["ttft_seconds"]) for row in pairs
            ),
            "decode_seconds": ar_decode_seconds,
            "decode_tok_s_weighted": ar_decode_tok_s,
            "total_seconds": ar_total_seconds,
            "e2e_output_tok_s_weighted": _ratio(output_tokens, ar_total_seconds),
        },
        "dflash": {
            "ttft_seconds": spec_ttft_seconds,
            "ttft_median_seconds": statistics.median(
                float(row["raw"]["dflash"]["ttft_seconds"]) for row in pairs
            ),
            "decode_seconds": spec_decode_seconds,
            "decode_tok_s_weighted": spec_decode_tok_s,
            "total_seconds": spec_total_seconds,
            "e2e_output_tok_s_weighted": _ratio(output_tokens, spec_total_seconds),
            "cycles": cycles,
            "cycle_wall_seconds": sum(
                float(row["raw"]["dflash"]["cycle_wall_seconds"]) for row in pairs
            ),
            "cycle_wall_mean_seconds": _ratio(
                sum(float(row["raw"]["dflash"]["cycle_wall_seconds"]) for row in pairs),
                cycles,
            ),
            "proposal_seconds": sum(
                float(row["raw"]["dflash"]["proposal_seconds"]) for row in pairs
            ),
            "target_verify_seconds": sum(
                float(row["raw"]["dflash"]["target_verify_seconds"]) for row in pairs
            ),
            "post_verify_residual_seconds": sum(
                float(row["raw"]["dflash"]["post_verify_residual_seconds"]) for row in pairs
            ),
            "accepted_draft_tokens": accepted,
            "draft_tokens_proposed": proposed,
            "draft_acceptance": _ratio(accepted, proposed),
            "accepted_per_output": _ratio(accepted, decode_tokens),
            "target_verify_rows": target_rows,
            "target_verify_rows_per_output": _ratio(target_rows, decode_tokens),
        },
        "comparison": {
            "decode_speedup_vs_true_ar": _ratio(spec_decode_tok_s, ar_decode_tok_s),
            "ttft_speedup_vs_true_ar": _ratio(ar_ttft_seconds, spec_ttft_seconds),
            "e2e_speedup_vs_true_ar": _ratio(
                _ratio(output_tokens, spec_total_seconds),
                _ratio(output_tokens, ar_total_seconds),
            ),
        },
    }


def _correctness(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_rows = []
    for row in pairs:
        ar = row["raw"]["ar"]
        spec = row["raw"]["dflash"]
        exact = ar["generated_ids"] == spec["generated_ids"]
        finite = bool(
            ar["finite_logits"] and spec["finite_draft_logits"] and spec["finite_verify_logits"]
        )
        aligned = bool(
            ar["state_aligned"] and spec["state_aligned"] and spec["all_verifier_addresses_stable"]
        )
        pair_rows.append(
            {
                "prompt_id": row["prompt"]["id"],
                "repetition": row["repetition"],
                "exact_match_ar": exact,
                "finite_all_logits": finite,
                "state_aligned": aligned,
                "pass": bool(exact and finite and aligned),
            }
        )
    deterministic = True
    for mode in ("ar", "dflash"):
        for prompt_id in {str(row["prompt"]["id"]) for row in pairs}:
            hashes = {
                str(row["raw"][mode]["generated_ids_sha256"])
                for row in pairs
                if row["prompt"]["id"] == prompt_id
            }
            deterministic = deterministic and len(hashes) == 1
    return {
        "pass": bool(pair_rows and all(row["pass"] for row in pair_rows) and deterministic),
        "all_pairs_exact_finite_aligned": bool(pair_rows and all(row["pass"] for row in pair_rows)),
        "same_mode_repeat_deterministic": bool(deterministic),
        "pair_rows": pair_rows,
    }


def _promotion_gate(
    *,
    correctness_passed: bool,
    protocol_eligible: bool,
    full_metrics: dict[str, Any],
    heldout_metrics: dict[str, Any],
    category_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    speedup = float(full_metrics["comparison"]["decode_speedup_vs_true_ar"])
    heldout_speedup = float(
        heldout_metrics["comparison"]["decode_speedup_vs_true_ar"]
    )
    category_speedups = {
        category: float(metrics["comparison"]["decode_speedup_vs_true_ar"])
        for category, metrics in category_metrics.items()
    }
    failed: list[str] = []
    if not correctness_passed:
        failed.append("exact_finite_state_correctness")
    if not protocol_eligible:
        failed.append("canonical_protocol")
    if not speedup > 1.10:
        failed.append("full_suite_decode_speedup_gt_1p10")
    if heldout_speedup < 1.0:
        failed.append("heldout_decode_non_regression")
    for category, category_speedup in category_speedups.items():
        if category_speedup < 1.0:
            failed.append(f"category_{category}_decode_non_regression")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "full_suite_decode_speedup_vs_true_ar": speedup,
        "heldout_decode_speedup_vs_true_ar": heldout_speedup,
        "category_decode_speedups_vs_true_ar": category_speedups,
        "policy": (
            "canonical 10-prompt train+heldout suite, exact/finite/state-aligned "
            "same-session output, weighted full-suite DFlash decode speedup >1.10x "
            "true AR, and no heldout/category decode regression"
        ),
    }


def _warmup(
    target: LagunaGGUFResidentSession,
    drafter: LagunaDFlashResidentDrafter,
    cycle: LagunaDFlashResidentCycle,
    prompt: dict[str, Any],
    *,
    output_tokens: int,
) -> None:
    _reset_request(target, drafter)
    _run_ar(target, prompt, output_tokens=output_tokens)
    _reset_request(target, drafter)
    _run_dflash(
        target,
        drafter,
        cycle,
        prompt,
        output_tokens=output_tokens,
    )
    _reset_request(target, drafter)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_size != RETAINED_CHUNK_SIZE:
        raise ValueError(
            f"retained Laguna DFlash economics requires chunk size {RETAINED_CHUNK_SIZE}"
        )
    if args.candidate_budget != ADMITTED_BUDGET:
        raise ValueError(f"retained Laguna DFlash economics requires admitted B{ADMITTED_BUDGET}")
    if args.output_tokens != DEFAULT_OUTPUT_TOKENS:
        raise ValueError(
            f"retained Laguna DFlash economics requires {DEFAULT_OUTPUT_TOKENS} output tokens"
        )
    if args.repetitions < DEFAULT_REPETITIONS:
        raise ValueError(
            f"retained Laguna DFlash economics requires at least {DEFAULT_REPETITIONS} repetitions"
        )
    if args.warmup_output_tokens <= args.candidate_budget:
        raise ValueError("warmup output tokens must exceed the candidate budget")
    if args.safety_reserve_gib <= 0.0:
        raise ValueError("--safety-reserve-gib must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna target model not found: {args.model}")
    if not args.drafter.is_dir():
        raise FileNotFoundError(f"Laguna DFlash safetensors snapshot not found: {args.drafter}")
    if not args.model_sha256 or not args.drafter_sha256 or not args.drafter_revision:
        raise ValueError("retained Laguna DFlash economics requires model fingerprints")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna DFlash benchmark requires a clean tracked worktree")

    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    if (
        max(prompt["prompt_tokens"] for prompt in prompts) + args.output_tokens
        > args.context_length
    ):
        raise ValueError("prompt/output shape exceeds admitted context")
    prompt_ids = {str(prompt["id"]) for prompt in prompts}
    if not DEFAULT_HELDOUT_IDS < prompt_ids:
        raise ValueError("canonical Laguna DFlash heldout IDs are incomplete")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant=args.quant_label,
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_dflash_category_economics_b4",
        timing_protocol="same_process_true_ar_vs_dflash_fixed_horizon",
        warmups=2,
        repetitions=args.repetitions,
    )
    compiler_version = _compiler_version(args.compiler_version_file)
    reset_memory_stats()
    tracked_before = memory_stats()
    runtime = None
    rows: list[dict[str, Any]] = []
    target_load_seconds = 0.0
    drafter_load_seconds = 0.0
    target_resident_nbytes = 0
    drafter_resident_nbytes = 0
    target_global_prefill_variant = ""
    target_swa_prefill_variant = ""
    allocated_after_load_bytes = 0
    oracle_gate: dict[str, Any] = {}
    process_started = time.perf_counter()
    target_started = time.perf_counter()
    with LagunaGGUFResidentSession(
        args.model,
        context_length=args.context_length,
        backend=args.backend,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        safety_reserve_nbytes=int(args.safety_reserve_gib * 2**30),
        progress=_progress,
        repacked_cache=None if args.direct_gguf else args.repacked_cache,
        model_sha256=args.model_sha256,
        prefill_chunk_size=args.chunk_size,
        iq3_selected_down_tile=args.iq3_selected_down_tile,
    ) as target:
        runtime = target.runtime
        target_load_seconds = time.perf_counter() - target_started
        (
            target_global_prefill_variant,
            target_swa_prefill_variant,
        ) = _resolved_target_prefill_variants(target)
        oracle_gate = _oracle_gate(target, args)
        drafter_started = time.perf_counter()
        with LagunaDFlashResidentDrafter(
            target,
            args.drafter,
            candidate_budget=args.candidate_budget,
            top_k=1,
            max_append_rows=args.candidate_budget + 1,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        ) as drafter:
            drafter_load_seconds = time.perf_counter() - drafter_started
            with LagunaDFlashResidentCycle(target, drafter) as cycle:
                target_resident_nbytes = target.resident_nbytes
                drafter_resident_nbytes = drafter.resident_nbytes
                allocated_after_load_bytes = int(memory_stats()["current_allocated_bytes"])
                _warmup(
                    target,
                    drafter,
                    cycle,
                    prompts[0],
                    output_tokens=args.warmup_output_tokens,
                )
                for repetition in range(args.repetitions):
                    for prompt_index, prompt in enumerate(prompts):
                        modes = (
                            ("ar", "dflash")
                            if (repetition + prompt_index) % 2 == 0
                            else ("dflash", "ar")
                        )
                        for mode in modes:
                            _reset_request(target, drafter)
                            if mode == "ar":
                                row = _run_ar(
                                    target,
                                    prompt,
                                    output_tokens=args.output_tokens,
                                )
                            else:
                                row = _run_dflash(
                                    target,
                                    drafter,
                                    cycle,
                                    prompt,
                                    output_tokens=args.output_tokens,
                                )
                            row["repetition"] = repetition
                            rows.append(row)
                            print(
                                f"rep={repetition} prompt={prompt['id']} mode={mode} "
                                f"ttft={row['ttft_seconds']:.3f}s "
                                f"decode={row['decode_tok_s']:.2f} tok/s",
                                file=sys.stderr,
                                flush=True,
                            )
                _reset_request(target, drafter)
                inside_memory = memory_stats()
    tracked_after = memory_stats()
    if runtime is None:
        raise RuntimeError("Laguna runtime was not initialized")
    gpu_free_after, gpu_total_bytes = runtime.mem_get_info()
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    pairs = _pair_rows(
        rows,
        candidate_budget=args.candidate_budget,
        peak_allocated_bytes=int(inside_memory["peak_allocated_bytes"]),
        allocated_after_load_bytes=allocated_after_load_bytes,
        backend=args.backend,
    )
    suite_correctness = _correctness(pairs)
    correctness_passed = bool(oracle_gate.get("pass") and suite_correctness["pass"] and recovered)
    full_metrics = _aggregate_scope(pairs)
    train_pairs = [row for row in pairs if row["split"] == "train"]
    heldout_pairs = [row for row in pairs if row["split"] == "heldout"]
    split_metrics = {
        "full": full_metrics,
        "train": _aggregate_scope(train_pairs),
        "heldout": _aggregate_scope(heldout_pairs),
    }
    category_metrics = {
        category: _aggregate_scope([row for row in pairs if row["prompt"]["category"] == category])
        for category in sorted(EXPECTED_CATEGORIES)
    }
    protocol_eligible = bool(
        len(prompts) == EXPECTED_PROMPT_COUNT
        and {prompt["category"] for prompt in prompts} == EXPECTED_CATEGORIES
        and {row["prompt"]["id"] for row in heldout_pairs} == DEFAULT_HELDOUT_IDS
        and args.candidate_budget == ADMITTED_BUDGET
        and args.output_tokens == DEFAULT_OUTPUT_TOKENS
        and args.repetitions >= DEFAULT_REPETITIONS
    )
    promotion = _promotion_gate(
        correctness_passed=correctness_passed,
        protocol_eligible=protocol_eligible,
        full_metrics=full_metrics,
        heldout_metrics=split_metrics["heldout"],
        category_metrics=category_metrics,
    )
    if not correctness_passed:
        decision_reason = "same-session exact/finite/state/oracle correctness gate failed"
    elif not promotion["pass"]:
        decision_reason = (
            "correctness retained, but DFlash failed promotion checks: "
            + ", ".join(promotion["failed_checks"])
        )
    else:
        decision_reason = (
            "exact full-suite DFlash decode exceeded true AR by >1.10x without "
            "heldout/category decode regression"
        )
    status = (
        "accepted"
        if promotion["pass"]
        else ("diagnostic_retained" if correctness_passed else "rejected_correctness")
    )
    models = SpeculativeBenchmarkModels(
        target_name="poolside/Laguna-S-2.1-GGUF",
        target_path=str(args.model.resolve()),
        target_revision=args.model_sha256,
        target_quant=args.quant_label,
        drafter_name="poolside/Laguna-S-2.1-DFlash",
        drafter_path=str(args.drafter.resolve()),
        drafter_revision=args.drafter_revision,
        drafter_dtype="bf16 safetensors",
    )
    artifact = build_speculative_artifact(
        run_tag="laguna-s21-dflash-category-economics-b4",
        summary="Laguna DFlash B4 economics against true same-process target AR",
        rows=pairs,
        models=models,
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware={
            "gpu": provenance["device_name"],
            "arch": args.backend.removeprefix("hip_"),
            "backend": args.backend,
            "hip_total_bytes": gpu_total_bytes,
            "gpu_free_after": gpu_free_after,
        },
        software={
            "hipengine_revision": repo["revision"],
            "tracked_clean": repo["tracked_clean"],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hipcc_version": compiler_version,
        },
        workload={
            "shape": "single_request_fixed_horizon_speculative_decode",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(args.prompts.read_bytes()),
            "prompt_count": len(prompts),
            "categories": sorted(EXPECTED_CATEGORIES),
            "train_prompt_ids": sorted(prompt_ids - DEFAULT_HELDOUT_IDS),
            "heldout_prompt_ids": sorted(DEFAULT_HELDOUT_IDS),
            "output_tokens_including_first": args.output_tokens,
            "timed_decode_output_tokens_per_run": args.output_tokens - 1,
            "candidate_budget": args.candidate_budget,
            "target_iq3_selected_down_tile": args.iq3_selected_down_tile,
            "repetitions": args.repetitions,
            "context_length": args.context_length,
            "prefill_chunk_size": args.chunk_size,
            "target_direct_gguf": bool(args.direct_gguf),
            "target_safety_reserve_gib": float(args.safety_reserve_gib),
            "target_global_prefill_variant": target_global_prefill_variant,
            "target_swa_prefill_variant": target_swa_prefill_variant,
            "sampling": "greedy argmax fixed horizon after stop",
            "same_session_ar_required": True,
            "speed_promotion_gate": (
                ">1.10x full-suite true AR, exact/finite/state correctness, and "
                "no heldout/category decode regression"
            ),
        },
        commands={
            "benchmark": [str(Path(sys.executable).resolve()), *sys.argv],
            "environment": {
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            },
        },
        notes=(
            "True AR uses exact bulk prefill; DFlash D1 context seeding is the current serial capture path.",
            "Decode timing starts after the synchronized first token and covers exactly output_tokens-1 visible tokens.",
            "Every final DFlash append is synchronized inside the measured cycle wall.",
            "No graph replay is used; model load is excluded from TTFT/decode/e2e rates.",
        ),
        decision_reason=decision_reason,
    )
    artifact.update(
        {
            "kind": "hipengine_laguna_dflash_category_economics",
            "pass": correctness_passed,
            "performance_claim": bool(promotion["pass"]),
            "performance_claim_scope": (
                f"Laguna S 2.1 {args.quant_label} + matched DFlash BF16 B4, c=1, canonical "
                "10-prompt four-category train+heldout suite, fixed 32-token horizon"
            ),
            "provenance": provenance,
            "repo": repo,
            "protocol": {
                "true_autoregressive_path": True,
                "same_prompt_suite": True,
                "same_process": True,
                "same_timing_protocol": True,
                "mode_order": "alternating AR/DFlash by repetition plus prompt index",
                "warmups": {"ar": 1, "dflash": 1},
                "ttft_scope": (
                    "prefill start through synchronized first-token argmax; AR bulk vs "
                    "current DFlash serial hidden-capture seed"
                ),
                "decode_scope": (
                    "post-TTFT through 31 synchronized visible outputs; DFlash final "
                    "accepted-prefix append included"
                ),
                "e2e_scope": "TTFT plus decode; model loading excluded",
                "protocol_eligible": protocol_eligible,
            },
            "correctness": {
                "pass": correctness_passed,
                "poolside_oracle": oracle_gate,
                "same_session_suite": suite_correctness,
                "tracked_returned_to_baseline": recovered,
            },
            "promotion_gate": promotion,
            "splits": split_metrics,
            "categories": category_metrics,
            "prompt_runs": pairs,
            "load": {
                "target_seconds_excluded": target_load_seconds,
                "drafter_seconds_excluded": drafter_load_seconds,
                "process_seconds": time.perf_counter() - process_started,
            },
            "memory": {
                "target_resident_nbytes": target_resident_nbytes,
                "drafter_resident_nbytes": drafter_resident_nbytes,
                "combined_resident_nbytes": (target_resident_nbytes + drafter_resident_nbytes),
                "allocated_after_load_bytes": allocated_after_load_bytes,
                "tracked_before": tracked_before,
                "tracked_inside": inside_memory,
                "tracked_after": tracked_after,
                "tracked_returned_to_baseline": recovered,
                "gpu_free_after": gpu_free_after,
                "hip_total_bytes": gpu_total_bytes,
            },
        }
    )
    artifact["decision"]["promotion_rule"] = promotion["policy"]
    return artifact


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
