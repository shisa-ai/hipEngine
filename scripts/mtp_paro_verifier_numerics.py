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
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from hipengine.speculative.mtp_native import NativeMtpChainProposer, NativeMtpW8A16Head
from scripts.mtp_prompt_suite_economics import _load_prompt_encoder, _load_prompt_suite
from scripts.quant_quality.metrics import per_row_metrics

_ROUTE_FLAGS = (
    "HIPENGINE_GDN_TLOOP_C1_EXACT",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX",
)
_STRICT_FLAGS = {
    "HIPENGINE_GDN_TLOOP_C1_EXACT": "1",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "1",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": "1",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX": "0",
}
_FAST_FLAGS = {
    "HIPENGINE_GDN_TLOOP_C1_EXACT": "0",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "0",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": "0",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX": "0",
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
        DeviceBuffer(int(session.batch_lm_logits.ptr), int(host.nbytes)),
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


def _summary(values: np.ndarray, top1: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(values.size),
        "mean_kl": float(np.mean(values)),
        "p95_kl": float(np.percentile(values, 95.0)),
        "p99_kl": float(np.percentile(values, 99.0)),
        "max_kl": float(np.max(values)),
        "top1_agreement": float(np.mean(top1)),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--prompt-render", choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"), default="qwen_chat_thinking_off")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        model=args.model,
        prompts_file=args.prompts_file,
        prompt_name=args.prompt_name,
        prompt_render=args.prompt_render,
        decode_tokens=int(args.decode_tokens),
        backend=args.backend,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["aggregate"], "first_decision_mismatch": result["first_decision_mismatch"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
