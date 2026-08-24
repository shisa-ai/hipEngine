#!/usr/bin/env python3
"""Teacher-forced strict-versus-fast PARO MTP verifier numerical gate.

Runs strict and candidate target sessions on the same prompt and strict-owned
B1 proposal/commit schedule. Full-vocabulary verifier logits are compared for
every aligned c2 row. Candidate commits are forced back to the strict-selected
row after each comparison so downstream contexts remain teacher-forced even if
a discrete candidate decision differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.execution_profiles import (
    VariantSelection,
    build_variant_manifest,
    manifest_sha256,
    resolve_runtime_profile,
)
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from hipengine.speculative.mtp_native import NativeMtpChainProposer, NativeMtpW8A16Head
from hipengine.speculative.paro_mtp_profiles import (
    FAST_VERIFIER_CANDIDATE_VARIANT,
    PARO_MTP_BACKEND,
    PARO_MTP_MODEL,
    PARO_MTP_MODEL_QUANT,
    PARO_MTP_REGISTRY_QUANT,
    PROPOSER_LAYER,
    STRICT_VERIFIER_VARIANT,
    TARGET_CONTRACT_VARIANT,
    VERIFIER_LAYER,
)
from scripts.mtp_prompt_suite_economics import _load_prompt_encoder, _load_prompt_suite
from scripts.quant_quality.metrics import per_row_metrics

_ROUTE_FLAGS = (
    "HIPENGINE_GDN_TLOOP_C1_EXACT",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX",
    "HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD",
)
_STRICT_FLAGS = {
    "HIPENGINE_GDN_TLOOP_C1_EXACT": "1",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "1",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": "1",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX": "0",
    "HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD": "off",
}
_FAST_FLAGS = {
    "HIPENGINE_GDN_TLOOP_C1_EXACT": "0",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "0",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": "0",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX": "0",
    "HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD": "off",
}
_THRESHOLDS = {
    "mean_kl_max": 1.0e-3,
    "p95_kl_max": 5.0e-3,
    "p99_kl_max": 2.0e-2,
    "max_kl_max": 5.0e-2,
    "top1_min": 0.99,
    "per_scope_top1_min": 0.97,
}


def _set_route(flags: dict[str, str]) -> None:
    os.environ.update(flags)


def _review_manifests() -> dict[str, Any]:
    strict = resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile="strict",
    )
    production = resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile="production",
    )
    candidate = build_variant_manifest(
        profile="production",
        backend=PARO_MTP_BACKEND,
        model=PARO_MTP_MODEL,
        quant=PARO_MTP_MODEL_QUANT,
        kv_policy="paged_bf16_kv_live_spans",
        graph_policy="off_b1_fixed_chain",
        selections=(
            VariantSelection(
                layer=PROPOSER_LAYER,
                scope="b1_graph_off_fixed_chain",
                selected_variant=TARGET_CONTRACT_VARIANT,
                strict_fallback_variant=TARGET_CONTRACT_VARIANT,
                registry_quant=PARO_MTP_REGISTRY_QUANT,
            ),
            VariantSelection(
                layer=VERIFIER_LAYER,
                scope="b1_graph_off_fixed_chain",
                selected_variant=FAST_VERIFIER_CANDIDATE_VARIANT,
                strict_fallback_variant=STRICT_VERIFIER_VARIANT,
                registry_quant=PARO_MTP_REGISTRY_QUANT,
            ),
        ),
    )
    return {
        "registered_strict_sha256": strict.manifest_sha256,
        "registered_production_sha256": production.manifest_sha256,
        "candidate_review_sha256": manifest_sha256(candidate),
        "candidate_review_manifest": candidate,
        "candidate_registered_but_uncertified": True,
        "candidate_selected_by_production": False,
    }


def _target_batch(root: int, context: int, candidate: int) -> TargetVerifyBatch:
    return TargetVerifyBatch.from_draft(
        compile_mtp_chain(
            [MtpDraftRequest(request_id=0, root_position=context, candidate_tokens=(candidate,), active_count=1)],
            candidate_budget=1,
        ),
        root_tokens=(root,),
        root_positions=(context,),
    )


def _copy_logits(session: Qwen35ParoResidentSession, *, rows: int = 2) -> np.ndarray:
    host = np.empty((rows, int(session.vocab_size)), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(session.verify_lm_logits.ptr), int(host.nbytes)),
        host.nbytes,
        runtime=session.runtime,
    )
    return host


def _prefill(
    session: Qwen35ParoResidentSession,
    tokens: Sequence[int],
    *,
    capture: Tensor | None = None,
) -> int:
    result = None
    for position, token in enumerate(tokens):
        if capture is None:
            result = session.step(int(token), position=position, sample=position == len(tokens) - 1)
        else:
            result = session.step_with_hidden_taps(
                int(token),
                position=position,
                capture_layer_ids=(int(session.layer_limit) - 1,),
                capture_hidden_concat=capture,
                capture_row=position,
                sample=position == len(tokens) - 1,
                capture_final_hidden_bf16=capture,
            )
    if result is None:
        raise RuntimeError("prompt prefill produced no root token")
    return int(result.token_id)


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    if successes < 0 or successes > total:
        raise ValueError("Wilson successes must be in [0, total]")
    proportion = float(successes) / float(total)
    z2 = float(z) * float(z)
    denominator = 1.0 + z2 / float(total)
    center = (proportion + z2 / (2.0 * float(total))) / denominator
    radius = (
        float(z)
        * math.sqrt(
            proportion * (1.0 - proportion) / float(total)
            + z2 / (4.0 * float(total) * float(total))
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _summary(values: np.ndarray, top1: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or top1.size != values.size:
        raise ValueError("summary needs non-empty aligned KL/top-1 vectors")
    matches = int(np.count_nonzero(top1))
    low, high = _wilson_interval(matches, int(values.size))
    return {
        "rows": int(values.size),
        "mean_kl": float(np.mean(values)),
        "p95_kl": float(np.percentile(values, 95.0)),
        "p99_kl": float(np.percentile(values, 99.0)),
        "max_kl": float(np.max(values)),
        "top1_agreement": float(np.mean(top1)),
        "top1_matches": matches,
        "top1_mismatches": int(values.size) - matches,
        "top1_wilson95_low": low,
        "top1_wilson95_high": high,
    }


def _stable_topk(row: np.ndarray, k: int) -> list[int]:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    if k <= 0 or k > values.size:
        raise ValueError("top-k must be in [1, vocab]")
    candidate_ids = np.argpartition(values, -k)[-k:]
    ordered = candidate_ids[np.lexsort((candidate_ids, -values[candidate_ids]))]
    return [int(token) for token in ordered]


def _token_probability(row: np.ndarray, token_id: int) -> float:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    maximum = float(np.max(values))
    denominator = float(np.exp(values - maximum).sum())
    return float(math.exp(float(values[int(token_id)]) - maximum) / denominator)


def _row_review_diagnostic(
    strict_row: np.ndarray,
    candidate_row: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    strict_values = np.asarray(strict_row, dtype=np.float32).reshape(-1)
    candidate_values = np.asarray(candidate_row, dtype=np.float32).reshape(-1)
    if strict_values.shape != candidate_values.shape or strict_values.size < 2:
        raise ValueError("row diagnostics need aligned vocab vectors with at least two tokens")
    strict_top = _stable_topk(strict_values, max(2, top_k))
    candidate_top = _stable_topk(candidate_values, max(2, top_k))
    strict_top1 = strict_top[0]
    candidate_top1 = candidate_top[0]
    strict_margin = float(strict_values[strict_top1] - strict_values[strict_top[1]])
    candidate_margin = float(
        candidate_values[candidate_top1] - candidate_values[candidate_top[1]]
    )
    return {
        "strict_topk": strict_top[:top_k],
        "candidate_topk": candidate_top[:top_k],
        "strict_margin": strict_margin,
        "candidate_margin": candidate_margin,
        "strict_top1_candidate_rank": int(
            1 + np.count_nonzero(candidate_values > candidate_values[strict_top1])
        ),
        "candidate_top1_strict_rank": int(
            1 + np.count_nonzero(strict_values > strict_values[candidate_top1])
        ),
        "strict_gap_to_candidate_top1": float(
            strict_values[strict_top1] - strict_values[candidate_top1]
        ),
        "candidate_gap_to_strict_top1": float(
            candidate_values[candidate_top1] - candidate_values[strict_top1]
        ),
        "strict_top1_probability": _token_probability(strict_values, strict_top1),
        "candidate_probability_of_strict_top1": _token_probability(
            candidate_values, strict_top1
        ),
        "strict_probability_of_candidate_top1": _token_probability(
            strict_values, candidate_top1
        ),
        "candidate_top1_probability": _token_probability(
            candidate_values, candidate_top1
        ),
    }


def _scope_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("scope summaries need at least one row")
    result: dict[str, dict[str, Any]] = {}
    for dimension in (
        "category",
        "shape",
        "transition",
        "row_role",
        "decision_role",
        "strict_selected_for_commit",
    ):
        groups: dict[str, Any] = {}
        for value in sorted({str(row[dimension]) for row in rows}):
            selected = [row for row in rows if str(row[dimension]) == value]
            summary = _summary(
                np.asarray([row["kl"] for row in selected], dtype=np.float64),
                np.asarray([row["top1_equal"] for row in selected], dtype=np.bool_),
            )
            summary["top5_overlap_mean"] = float(
                np.mean([row["top5_overlap"] for row in selected])
            )
            summary["strict_margin_min"] = float(
                min(row["strict_margin"] for row in selected)
            )
            summary["passed"] = bool(
                summary["mean_kl"] <= _THRESHOLDS["mean_kl_max"]
                and summary["p95_kl"] <= _THRESHOLDS["p95_kl_max"]
                and summary["p99_kl"] <= _THRESHOLDS["p99_kl_max"]
                and summary["max_kl"] <= _THRESHOLDS["max_kl_max"]
                and summary["top1_agreement"] >= _THRESHOLDS["per_scope_top1_min"]
            )
            groups[value] = summary
        result[dimension] = groups
    return result


def run(
    *,
    model: Path,
    prompts_file: Path,
    prompt_name: str,
    prompt_render: str,
    decode_tokens: int,
    backend: str,
) -> dict[str, Any]:
    suite = _load_prompt_suite(prompts_file)
    prompt = next((row for row in suite["prompts"] if row["name"] == prompt_name), None)
    if prompt is None:
        raise ValueError(f"unknown prompt {prompt_name!r}")
    encoder = _load_prompt_encoder(model, prompt_render)
    prompt_tokens = [int(token) for token in encoder.encode(prompt["prompt"]).token_ids]
    if decode_tokens < 2:
        raise ValueError("decode_tokens must be at least 2")

    saved_env = {key: os.environ.get(key) for key in _ROUTE_FLAGS}
    started = time.perf_counter()
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + decode_tokens + 8
    capture_buf = None
    row_metrics: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    generated: list[int] = []
    decision_mismatch = None
    try:
        _set_route(_STRICT_FLAGS)
        with Qwen35ParoResidentSession(
            runner, max_sequence_length=max_sequence, max_batch_size=2
        ) as strict_session:
            _set_route(_FAST_FLAGS)
            with Qwen35ParoResidentSession(
                runner, max_sequence_length=max_sequence, max_batch_size=2
            ) as candidate_session:
                hidden = int(strict_session.config.hidden_size)
                capture_buf = malloc(len(prompt_tokens) * hidden * DType.BF16.itemsize, runtime=strict_session.runtime)
                capture = Tensor.from_handle(
                    capture_buf.ptr,
                    (len(prompt_tokens), hidden),
                    DType.BF16,
                    Device("hip", 0),
                )
                no_capture = Tensor.from_handle(0, (2, 0), DType.BF16, Device("hip", 0))
                _set_route(_STRICT_FLAGS)
                strict_root = _prefill(strict_session, prompt_tokens, capture=capture)
                _set_route(_FAST_FLAGS)
                candidate_root = _prefill(candidate_session, prompt_tokens)
                scoring_head = NativeMtpW8A16Head(
                    weight_int8_ptr=int(strict_session.lm_head_weight.tensor.ptr),
                    scale_f32_ptr=int(strict_session.lm_head_scale.tensor.ptr),
                    vocab_size=int(strict_session.vocab_size),
                    threads=int(strict_session.lm_head_threads),
                    owner=strict_session,
                )
                _set_route(_STRICT_FLAGS)
                with NativeMtpChainProposer(
                    model,
                    max_positions=max_sequence,
                    max_mtp_tokens=len(prompt_tokens) + 2 * decode_tokens + 8,
                    runtime=strict_session.runtime,
                    compiler_version=strict_session.compiler_version,
                    scoring_head=scoring_head,
                ) as proposer:
                    proposer.prefill_from_target_hidden_rows(
                        prompt_tokens,
                        capture_base_ptr=capture_buf.ptr,
                        seed_token=strict_root,
                        read_expert_topk=False,
                        read_lm_head_value=False,
                    )
                    context = len(prompt_tokens)
                    root = strict_root
                    cycle = 0
                    while len(generated) < decode_tokens:
                        cycle += 1
                        candidate = int(proposer.current.token)
                        batch = _target_batch(root, context, candidate)
                        _set_route(_STRICT_FLAGS)
                        strict = strict_session.verify_chain_bulk_and_commit(
                            batch,
                            base_slot=0,
                            capture_layer_ids=(),
                            capture_hidden_concat=no_capture,
                            capture_row_start=0,
                            chain_attn_mode="c1_loop",
                            graph_mode="off",
                            canonicalize_after=False,
                        )
                        strict_logits = _copy_logits(strict_session)
                        _set_route(_FAST_FLAGS)
                        fast = candidate_session.verify_chain_bulk_and_commit(
                            batch,
                            base_slot=0,
                            capture_layer_ids=(),
                            capture_hidden_concat=no_capture,
                            capture_row_start=0,
                            chain_attn_mode="decode_batched",
                            graph_mode="off",
                            canonicalize_after=False,
                        )
                        fast_logits = _copy_logits(candidate_session)
                        labels = np.argmax(strict_logits, axis=1).astype(np.int64)
                        metrics = per_row_metrics(strict_logits, fast_logits, labels, top_k=5)
                        for row in range(2):
                            row_metrics.append(
                                {
                                    "cycle": cycle,
                                    "row": row,
                                    "category": str(prompt.get("category", "unknown")),
                                    "shape": "c2_b1",
                                    "transition": "prefill_to_verify" if cycle == 1 else "verify_to_verify",
                                    "strict_top1": int(strict.target_top1[row]),
                                    "candidate_top1": int(fast.target_top1[row]),
                                    "kl": float(metrics["kl_nats"][row]),
                                    "top1_equal": bool(metrics["top1_equal"][row]),
                                    "top5_overlap": float(metrics["topk_set_overlap"][row]),
                                    "max_abs_logit_delta": float(metrics["max_abs_logit_delta"][row]),
                                }
                            )
                        strict_bonus = int(strict.next_token if strict.next_token is not None else strict.target_top1[strict.accepted_count])
                        fast_bonus = int(fast.next_token if fast.next_token is not None else fast.target_top1[fast.accepted_count])
                        if decision_mismatch is None and (
                            int(fast.accepted_count) != int(strict.accepted_count)
                            or fast_bonus != strict_bonus
                        ):
                            decision_mismatch = {
                                "cycle": cycle,
                                "output_offset": len(generated),
                                "strict_accepted": int(strict.accepted_count),
                                "candidate_accepted": int(fast.accepted_count),
                                "strict_bonus": strict_bonus,
                                "candidate_bonus": fast_bonus,
                            }
                        committed = [root] + ([candidate] if int(strict.accepted_count) else [])
                        generated.extend(committed)
                        cycles.append(
                            {
                                "cycle": cycle,
                                "context": context,
                                "root": root,
                                "candidate": candidate,
                                "strict_accepted": int(strict.accepted_count),
                                "candidate_accepted": int(fast.accepted_count),
                                "strict_bonus": strict_bonus,
                                "candidate_bonus": fast_bonus,
                            }
                        )
                        # Candidate decisions never own the teacher-forced schedule.
                        if int(fast.commit_row) != int(strict.commit_row):
                            candidate_session._commit_bulk_linear_states(int(strict.commit_row), base_slot=0)
                            candidate_session._set_slot_position(int(strict.commit_position), slot=0)
                            candidate_session.runtime.device_synchronize()
                        if len(generated) >= decode_tokens:
                            break
                        if int(strict.accepted_count) >= 1:
                            _set_route(_STRICT_FLAGS)
                            proposer.advance_with_previous_hidden(
                                input_token=candidate,
                                position=proposer.position + 1,
                                need_result=False,
                                read_expert_topk=False,
                                read_lm_head_value=False,
                            )
                        proposer.advance_with_target_hidden(
                            input_token=strict_bonus,
                            target_hidden_ptr=int(strict.selected_target_hidden_ptr),
                            position=proposer.position + 1,
                            read_expert_topk=False,
                            read_lm_head_value=False,
                        )
                        context += len(committed)
                        root = strict_bonus
    finally:
        if capture_buf is not None:
            free(capture_buf)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    kl = np.asarray([row["kl"] for row in row_metrics], dtype=np.float64)
    top1 = np.asarray([row["top1_equal"] for row in row_metrics], dtype=np.bool_)
    aggregate = _summary(kl, top1)
    checks = {
        "prefill_root_equal": int(strict_root) == int(candidate_root),
        "mean_kl": aggregate["mean_kl"] <= _THRESHOLDS["mean_kl_max"],
        "p95_kl": aggregate["p95_kl"] <= _THRESHOLDS["p95_kl_max"],
        "p99_kl": aggregate["p99_kl"] <= _THRESHOLDS["p99_kl_max"],
        "max_kl": aggregate["max_kl"] <= _THRESHOLDS["max_kl_max"],
        "top1": aggregate["top1_agreement"] >= _THRESHOLDS["top1_min"],
        "task_decisions": decision_mismatch is None,
        "finite": bool(np.isfinite(kl).all()),
    }
    return {
        "schema": "hipengine.paro_mtp_verifier_numerics.v1",
        "status": "passed" if all(checks.values()) else "rejected",
        "performance_claim": False,
        "model": str(model),
        "backend": backend,
        "candidate": {
            "source_class": "T2",
            "chain_attn_mode": "decode_batched",
            "environment": _FAST_FLAGS,
            "strict_fallback": {"chain_attn_mode": "c1_loop", "environment": _STRICT_FLAGS},
        },
        "prompt": {
            "name": prompt_name,
            "category": prompt.get("category"),
            "split": prompt.get("split"),
            "render": prompt_render,
            "prompt_tokens": len(prompt_tokens),
            "decode_tokens": decode_tokens,
        },
        "teacher_forcing": "strict-owned proposals, commit rows, tokens, positions, and contexts",
        "thresholds": _THRESHOLDS,
        "aggregate": aggregate,
        "checks": checks,
        "first_decision_mismatch": decision_mismatch,
        "first_top1_mismatch": next((row for row in row_metrics if not row["top1_equal"]), None),
        "rows": row_metrics,
        "cycles": cycles,
        "seconds": time.perf_counter() - started,
    }


def run_sequential(
    *,
    model: Path,
    prompts_file: Path,
    prompt_name: str,
    prompt_render: str,
    decode_tokens: int,
    backend: str,
    prompt_tokens_file: Path | None = None,
) -> dict[str, Any]:
    """Run strict capture and fast replay sequentially to bound W7900 VRAM."""

    suite = _load_prompt_suite(prompts_file)
    prompt = next((row for row in suite["prompts"] if row["name"] == prompt_name), None)
    if prompt is None:
        raise ValueError(f"unknown prompt {prompt_name!r}")
    if prompt_tokens_file is None:
        encoder = _load_prompt_encoder(model, prompt_render)
        prompt_tokens = [int(token) for token in encoder.encode(prompt["prompt"]).token_ids]
    else:
        prompt_tokens = [
            int(token)
            for token in prompt_tokens_file.read_text(encoding="utf-8")
            .replace(",", " ")
            .split()
        ]
        if not prompt_tokens:
            raise ValueError("prompt token fixture is empty")
    if decode_tokens < 2:
        raise ValueError("decode_tokens must be at least 2")

    saved_env = {key: os.environ.get(key) for key in _ROUTE_FLAGS}
    started = time.perf_counter()
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + decode_tokens + 8
    schedule: list[dict[str, Any]] = []
    strict_logits_by_cycle: list[np.ndarray] = []
    strict_generated: list[int] = []
    row_metrics: list[dict[str, Any]] = []
    replay_cycles: list[dict[str, Any]] = []
    decision_mismatch = None
    strict_root = -1
    candidate_root = -1
    capture_buf = None
    try:
        print("[verifier-numerics] loading strict session", flush=True)
        _set_route(_STRICT_FLAGS)
        with Qwen35ParoResidentSession(
            runner, max_sequence_length=max_sequence, max_batch_size=2
        ) as strict_session:
            hidden = int(strict_session.config.hidden_size)
            capture_buf = malloc(
                len(prompt_tokens) * hidden * DType.BF16.itemsize,
                runtime=strict_session.runtime,
            )
            capture = Tensor.from_handle(
                capture_buf.ptr,
                (len(prompt_tokens), hidden),
                DType.BF16,
                Device("hip", 0),
            )
            no_capture = Tensor.from_handle(0, (2, 0), DType.BF16, Device("hip", 0))
            strict_root = _prefill(strict_session, prompt_tokens, capture=capture)
            scoring_head = NativeMtpW8A16Head(
                weight_int8_ptr=int(strict_session.lm_head_weight.tensor.ptr),
                scale_f32_ptr=int(strict_session.lm_head_scale.tensor.ptr),
                vocab_size=int(strict_session.vocab_size),
                threads=int(strict_session.lm_head_threads),
                owner=strict_session,
            )
            with NativeMtpChainProposer(
                model,
                max_positions=max_sequence,
                max_mtp_tokens=len(prompt_tokens) + 2 * decode_tokens + 8,
                runtime=strict_session.runtime,
                compiler_version=strict_session.compiler_version,
                scoring_head=scoring_head,
            ) as proposer:
                proposer.prefill_from_target_hidden_rows(
                    prompt_tokens,
                    capture_base_ptr=capture_buf.ptr,
                    seed_token=strict_root,
                    read_expert_topk=False,
                    read_lm_head_value=False,
                )
                context = len(prompt_tokens)
                root = strict_root
                cycle = 0
                while len(strict_generated) < decode_tokens:
                    cycle += 1
                    candidate = int(proposer.current.token)
                    batch = _target_batch(root, context, candidate)
                    strict = strict_session.verify_chain_bulk_and_commit(
                        batch,
                        base_slot=0,
                        capture_layer_ids=(),
                        capture_hidden_concat=no_capture,
                        capture_row_start=0,
                        chain_attn_mode="c1_loop",
                        graph_mode="off",
                        canonicalize_after=False,
                    )
                    strict_logits = _copy_logits(strict_session)
                    strict_logits_by_cycle.append(strict_logits)
                    strict_top1 = np.argmax(strict_logits, axis=1).astype(np.int64)
                    strict_bonus = int(
                        strict.next_token
                        if strict.next_token is not None
                        else strict_top1[strict.accepted_count]
                    )
                    committed = [root] + ([candidate] if int(strict.accepted_count) else [])
                    schedule.append(
                        {
                            "cycle": cycle,
                            "output_offset": len(strict_generated),
                            "context": context,
                            "root": root,
                            "candidate": candidate,
                            "strict_accepted": int(strict.accepted_count),
                            "strict_bonus": strict_bonus,
                            "strict_commit_row": int(strict.commit_row),
                            "strict_commit_position": int(strict.commit_position),
                            "strict_target_top1": [int(token) for token in strict_top1.tolist()],
                            "committed": committed,
                        }
                    )
                    strict_generated.extend(committed)
                    if len(strict_generated) >= decode_tokens:
                        break
                    if int(strict.accepted_count) >= 1:
                        proposer.advance_with_previous_hidden(
                            input_token=candidate,
                            position=proposer.position + 1,
                            need_result=False,
                            read_expert_topk=False,
                            read_lm_head_value=False,
                        )
                    proposer.advance_with_target_hidden(
                        input_token=strict_bonus,
                        target_hidden_ptr=int(strict.selected_target_hidden_ptr),
                        position=proposer.position + 1,
                        read_expert_topk=False,
                        read_lm_head_value=False,
                    )
                    context += len(committed)
                    root = strict_bonus
            free(capture_buf, runtime=strict_session.runtime)
            capture_buf = None
        print(
            f"[verifier-numerics] strict capture complete cycles={len(schedule)}; loading fast session",
            flush=True,
        )

        _set_route(_FAST_FLAGS)
        with Qwen35ParoResidentSession(
            runner, max_sequence_length=max_sequence, max_batch_size=2
        ) as candidate_session:
            no_capture = Tensor.from_handle(0, (2, 0), DType.BF16, Device("hip", 0))
            candidate_root = _prefill(candidate_session, prompt_tokens)
            output_offset = 0
            for record, strict_logits in zip(schedule, strict_logits_by_cycle, strict=True):
                batch = _target_batch(
                    int(record["root"]),
                    int(record["context"]),
                    int(record["candidate"]),
                )
                fast = candidate_session.verify_chain_bulk_and_commit(
                    batch,
                    base_slot=0,
                    capture_layer_ids=(),
                    capture_hidden_concat=no_capture,
                    capture_row_start=0,
                    chain_attn_mode="decode_batched",
                    graph_mode="off",
                    canonicalize_after=False,
                )
                fast_logits = _copy_logits(candidate_session)
                fast_top1 = np.argmax(fast_logits, axis=1).astype(np.int64)
                labels = np.argmax(strict_logits, axis=1).astype(np.int64)
                metrics = per_row_metrics(strict_logits, fast_logits, labels, top_k=5)
                fast_bonus = int(
                    fast.next_token
                    if fast.next_token is not None
                    else fast_top1[fast.accepted_count]
                )
                cycle_decision_mismatch = bool(
                    int(fast.accepted_count) != int(record["strict_accepted"])
                    or fast_bonus != int(record["strict_bonus"])
                )
                if decision_mismatch is None and cycle_decision_mismatch:
                    decision_mismatch = {
                        "cycle": int(record["cycle"]),
                        "output_offset": int(record["output_offset"]),
                        "strict_accepted": int(record["strict_accepted"]),
                        "candidate_accepted": int(fast.accepted_count),
                        "strict_bonus": int(record["strict_bonus"]),
                        "candidate_bonus": fast_bonus,
                    }
                for row in range(2):
                    diagnostic = _row_review_diagnostic(
                        strict_logits[row], fast_logits[row], top_k=5
                    )
                    row_metrics.append(
                        {
                            "cycle": int(record["cycle"]),
                            "output_offset": int(record["output_offset"]),
                            "row": row,
                            "row_role": "root" if row == 0 else "draft_candidate",
                            "decision_role": (
                                "draft_acceptance_or_reject_correction"
                                if row == 0
                                else "full_accept_bonus"
                            ),
                            "category": str(prompt.get("category", "unknown")),
                            "shape": "c2_b1",
                            "transition": (
                                "prefill_to_verify"
                                if int(record["cycle"]) == 1
                                else "verify_to_verify"
                            ),
                            "context": int(record["context"]),
                            "position": int(record["context"]) + row,
                            "strict_selected_for_commit": bool(
                                row == int(record["strict_commit_row"])
                            ),
                            "candidate_selected_for_commit": bool(
                                row == int(fast.commit_row)
                            ),
                            "cycle_task_decision_mismatch": cycle_decision_mismatch,
                            "strict_top1": int(record["strict_target_top1"][row]),
                            "candidate_top1": int(fast_top1[row]),
                            "kl": float(metrics["kl_nats"][row]),
                            "top1_equal": bool(metrics["top1_equal"][row]),
                            "top5_overlap": float(metrics["topk_set_overlap"][row]),
                            "max_abs_logit_delta": float(metrics["max_abs_logit_delta"][row]),
                            **diagnostic,
                        }
                    )
                if int(fast.commit_row) != int(record["strict_commit_row"]):
                    candidate_session._commit_bulk_linear_states(
                        int(record["strict_commit_row"]), base_slot=0
                    )
                    candidate_session._set_slot_position(
                        int(record["strict_commit_position"]), slot=0
                    )
                    candidate_session.runtime.device_synchronize()
                replay_cycles.append(
                    {
                        **record,
                        "candidate_accepted": int(fast.accepted_count),
                        "candidate_bonus": fast_bonus,
                        "candidate_commit_row": int(fast.commit_row),
                        "task_decision_mismatch": cycle_decision_mismatch,
                    }
                )
                output_offset += len(record["committed"])
        print("[verifier-numerics] fast replay complete", flush=True)
    finally:
        if capture_buf is not None:
            free(capture_buf)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    kl = np.asarray([row["kl"] for row in row_metrics], dtype=np.float64)
    top1 = np.asarray([row["top1_equal"] for row in row_metrics], dtype=np.bool_)
    aggregate = _summary(kl, top1)
    aggregate["sample_resolution"] = 1.0 / float(aggregate["rows"])
    aggregate["top5_overlap_mean"] = float(
        np.mean([row["top5_overlap"] for row in row_metrics])
    )
    aggregate["strict_margin_min"] = float(
        min(row["strict_margin"] for row in row_metrics)
    )
    scopes = _scope_summaries(row_metrics)
    scope_failures = [
        {"dimension": dimension, "value": value}
        for dimension, groups in scopes.items()
        for value, summary in groups.items()
        if not bool(summary["passed"])
    ]
    decision_mismatches = [
        {
            "cycle": int(cycle["cycle"]),
            "output_offset": int(cycle["output_offset"]),
            "strict_accepted": int(cycle["strict_accepted"]),
            "candidate_accepted": int(cycle["candidate_accepted"]),
            "strict_bonus": int(cycle["strict_bonus"]),
            "candidate_bonus": int(cycle["candidate_bonus"]),
        }
        for cycle in replay_cycles
        if bool(cycle["task_decision_mismatch"])
    ]
    top1_mismatch_rows = [row for row in row_metrics if not row["top1_equal"]]
    checks = {
        "prefill_root_equal": int(strict_root) == int(candidate_root),
        "mean_kl": aggregate["mean_kl"] <= _THRESHOLDS["mean_kl_max"],
        "p95_kl": aggregate["p95_kl"] <= _THRESHOLDS["p95_kl_max"],
        "p99_kl": aggregate["p99_kl"] <= _THRESHOLDS["p99_kl_max"],
        "max_kl": aggregate["max_kl"] <= _THRESHOLDS["max_kl_max"],
        "top1": aggregate["top1_agreement"] >= _THRESHOLDS["top1_min"],
        "per_scope": not scope_failures,
        "task_decisions": not decision_mismatches,
        "finite": bool(np.isfinite(kl).all()),
    }
    return {
        "schema": "hipengine.paro_mtp_verifier_numerics.v3",
        "status": "passed" if all(checks.values()) else "rejected",
        "performance_claim": False,
        "model": str(model),
        "backend": backend,
        "capture_mode": "sequential_strict_then_fast_replay",
        "manifests": _review_manifests(),
        "candidate": {
            "source_class": "T2",
            "chain_attn_mode": "decode_batched",
            "environment": _FAST_FLAGS,
            "strict_fallback": {
                "chain_attn_mode": "c1_loop",
                "environment": _STRICT_FLAGS,
            },
        },
        "prompt": {
            "name": prompt_name,
            "category": prompt.get("category"),
            "split": prompt.get("split"),
            "render": prompt_render,
            "prompt_tokens": len(prompt_tokens),
            "decode_tokens": decode_tokens,
        },
        "teacher_forcing": "strict-owned proposals, commit rows, tokens, positions, and contexts",
        "thresholds": _THRESHOLDS,
        "aggregate": aggregate,
        "scopes": scopes,
        "scope_failures": scope_failures,
        "checks": checks,
        "review": {
            "automatic_admission_threshold_unchanged": True,
            "one_mismatch_point_estimate": (
                float(aggregate["rows"] - 1) / float(aggregate["rows"])
            ),
            "top1_mismatch_rows": top1_mismatch_rows,
            "task_decision_mismatches": decision_mismatches,
        },
        "first_decision_mismatch": decision_mismatch,
        "first_top1_mismatch": top1_mismatch_rows[0] if top1_mismatch_rows else None,
        "rows": row_metrics,
        "cycles": replay_cycles,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--prompt-render", choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"), default="qwen_chat_thinking_off")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens-file", type=Path)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_sequential(
        model=args.model,
        prompts_file=args.prompts_file,
        prompt_name=args.prompt_name,
        prompt_render=args.prompt_render,
        decode_tokens=int(args.decode_tokens),
        backend=args.backend,
        prompt_tokens_file=args.prompt_tokens_file,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["aggregate"], "first_decision_mismatch": result["first_decision_mismatch"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
