#!/usr/bin/env python3
"""Correctness-first native MTP chain E2E smoke through the shared verifier.

This is an intermediate Task #41/#40 bridge.  It uses the native MTP proposal
chain to build candidate-only DraftBatch rows, then calls
Qwen35ParoResidentSession.verify_chain_bulk_and_commit.  To keep the first E2E
hook simple, proposal input hidden rows are copied back to host before invoking
``run_smoke(..., target_hidden_bits_override=...)``; therefore this script is not
a throughput benchmark and must not be promoted as a speed row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from scripts.mtp_native_decode_step_smoke import run_smoke as run_native_mtp_proposal

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


def _capture_tensor(buffer: DeviceBuffer, rows: int, hidden: int) -> Tensor:
    return Tensor.from_handle(buffer.ptr, (int(rows), int(hidden)), DType.BF16, Device("hip", 0))


def _read_capture_row(buffer: DeviceBuffer, row: int, hidden: int) -> np.ndarray:
    host = np.zeros((1, int(hidden)), dtype=np.uint16)
    offset = int(row) * int(hidden) * DType.BF16.itemsize
    view = DeviceBuffer(ptr=buffer.ptr + offset, nbytes=host.nbytes)
    copy_device_to_host(host_array_ptr(host), view, host.nbytes)
    return host


def _run_ar_baseline(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    backend: str,
) -> tuple[list[int], dict[str, Any]]:
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + int(decode_tokens) + 2
    started = time.perf_counter()
    generated: list[int] = []
    with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence) as session:
        next_result = None
        for pos, token in enumerate(prompt_tokens):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_tokens) - 1))
        if next_result is None:
            raise RuntimeError("prompt did not produce a root token")
        root = int(next_result.token_id)
        context = len(prompt_tokens)
        while len(generated) < int(decode_tokens):
            generated.append(root)
            if len(generated) >= int(decode_tokens):
                break
            next_result = session.step(root, position=context, sample=True)
            if next_result is None:
                raise RuntimeError("AR decode step produced no token")
            root = int(next_result.token_id)
            context += 1
    seconds = time.perf_counter() - started
    return generated, {"seconds": seconds, "tok_s": len(generated) / seconds if seconds > 0 else None}


def _run_spec_smoke(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    chain_attn_mode: str,
) -> tuple[list[int], dict[str, Any]]:
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + int(decode_tokens) + int(candidate_budget) + 4
    max_batch_size = int(candidate_budget) + 1
    generated: list[int] = []
    accepted_lengths: list[int] = []
    proposal_trace: list[dict[str, Any]] = []
    proposal_seconds = 0.0
    verify_seconds = 0.0
    target_forward_calls = 0
    capture_rows = max_sequence + int(candidate_budget) + 2
    capture_buf: DeviceBuffer | None = None
    started = time.perf_counter()
    with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence, max_batch_size=max_batch_size) as session:
        hidden = int(session.config.hidden_size)
        capture_layer_id = int(session.layer_limit) - 1
        capture_buf = malloc(capture_rows * hidden * DType.BF16.itemsize, runtime=session.runtime)
        capture = _capture_tensor(capture_buf, capture_rows, hidden)
        try:
            next_result = None
            for pos, token in enumerate(prompt_tokens):
                next_result = session.step_with_hidden_taps(
                    int(token),
                    position=pos,
                    capture_layer_ids=(capture_layer_id,),
                    capture_hidden_concat=capture,
                    capture_row=pos,
                    sample=(pos == len(prompt_tokens) - 1),
                )
            if next_result is None:
                raise RuntimeError("prompt did not produce a root token")
            root = int(next_result.token_id)
            context = len(prompt_tokens)
            previous_hidden_row = context - 1
            cycles = 0
            while len(generated) < int(decode_tokens):
                remaining = int(decode_tokens) - len(generated)
                active_budget = min(int(candidate_budget), max(0, remaining - 1))
                if active_budget <= 0:
                    step_result = session.step_with_hidden_taps(
                        root,
                        position=context,
                        capture_layer_ids=(capture_layer_id,),
                        capture_hidden_concat=capture,
                        capture_row=context,
                        sample=True,
                    )
                    if step_result is None:
                        raise RuntimeError("terminal AR step produced no root")
                    generated.append(root)
                    root = int(step_result.token_id)
                    previous_hidden_row = context
                    context += 1
                    continue
                cycles += 1
                target_hidden = _read_capture_row(capture_buf, previous_hidden_row, hidden)
                t_prop = time.perf_counter()
                proposal = run_native_mtp_proposal(
                    model,
                    root_token=root,
                    root_position=context,
                    draft_budget=active_budget,
                    torch_compare=False,
                    target_hidden_bits_override=target_hidden,
                )
                proposal_seconds += time.perf_counter() - t_prop
                candidates = [int(token) for token in proposal["candidate_tokens"][:active_budget]]
                draft = proposal["draft_batch"]
                target_batch = TargetVerifyBatch.from_draft(
                    compile_mtp_chain(
                        [MtpDraftRequest(request_id=0, root_position=context, candidate_tokens=tuple(candidates), active_count=len(candidates))],
                        candidate_budget=active_budget,
                    ),
                    root_tokens=(root,),
                    root_positions=(context,),
                )
                t_verify = time.perf_counter()
                verify = session.verify_chain_bulk_and_commit(
                    target_batch,
                    base_slot=0,
                    capture_layer_ids=(capture_layer_id,),
                    capture_hidden_concat=capture,
                    capture_row_start=context,
                    chain_attn_mode=chain_attn_mode,
                )
                verify_seconds += time.perf_counter() - t_verify
                target_forward_calls += int(verify.target_forward_calls)
                accepted = int(verify.accepted_count)
                accepted_lengths.append(accepted)
                accepted_tokens = candidates[:accepted]
                committed = [root, *accepted_tokens]
                generated.extend(committed)
                bonus = int(verify.next_token) if verify.next_token is not None else int(verify.target_top1[min(accepted, len(verify.target_top1) - 1)])
                if len(proposal_trace) < 16:
                    proposal_trace.append(
                        {
                            "cycle": cycles,
                            "root_position": context,
                            "root_token": root,
                            "draft_candidates": candidates,
                            "target_top1_path": list(map(int, verify.target_top1[: 1 + active_budget])),
                            "accepted": accepted,
                            "committed_tokens": committed,
                            "bonus_token": bonus,
                            "target_parent_rows": list(map(int, target_batch.parent_rows)),
                            "proposal_native_seconds": float(proposal["native_seconds"]),
                            "proposal_draft_batch": draft,
                        }
                    )
                previous_hidden_row = context + len(committed) - 1
                context += len(committed)
                root = bonus
        finally:
            if capture_buf is not None:
                free(capture_buf, runtime=session.runtime)
    seconds = time.perf_counter() - started
    return generated[: int(decode_tokens)], {
        "seconds": seconds,
        "tok_s": int(decode_tokens) / seconds if seconds > 0 else None,
        "proposal_seconds": proposal_seconds,
        "verify_seconds": verify_seconds,
        "accepted_lengths": accepted_lengths,
        "acceptance_rate": (sum(accepted_lengths) / (len(accepted_lengths) * int(candidate_budget))) if accepted_lengths and candidate_budget > 0 else 0.0,
        "proposal_trace_sample": proposal_trace,
        "target_forward_calls": target_forward_calls,
        "chain_attn_mode": chain_attn_mode,
        "note": "Correctness smoke only: proposal hidden rows are copied D2H and MTP weights are reloaded per proposal call.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    prompt_tokens = tuple(int(part.strip()) for part in str(args.prompt_tokens).split(",") if part.strip())
    if not prompt_tokens:
        raise ValueError("at least one prompt token is required")
    ar_tokens, ar = _run_ar_baseline(model, prompt_tokens, decode_tokens=int(args.decode_tokens), backend=str(args.backend))
    spec_tokens, spec = _run_spec_smoke(
        model,
        prompt_tokens,
        decode_tokens=int(args.decode_tokens),
        candidate_budget=int(args.candidate_budget),
        backend=str(args.backend),
        chain_attn_mode=str(args.chain_attn_mode),
    )
    return {
        "status": "passed" if spec_tokens == ar_tokens else "exact_ar_mismatch",
        "performance_claim": False,
        "model": str(model),
        "prompt_tokens": list(prompt_tokens),
        "decode_tokens": int(args.decode_tokens),
        "candidate_budget": int(args.candidate_budget),
        "ar_tokens": ar_tokens,
        "mtp_tokens": spec_tokens,
        "exact_ar_match": spec_tokens == ar_tokens,
        "ar": ar,
        "mtp": spec,
        "decision_reason": "Native MTP proposal rows reached verify_chain_bulk_and_commit and exact AR was checked. This is not a speed row because proposal hidden rows are copied to host and MTP weights are reloaded per proposal call.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default="151646")
    parser.add_argument("--decode-tokens", type=int, default=3)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched"), default="c1_loop")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "exact_ar_match": result["exact_ar_match"], "ar": result["ar_tokens"], "mtp": result["mtp_tokens"], "accepted": result["mtp"]["accepted_lengths"], "mtp_tok_s_diagnostic": result["mtp"]["tok_s"], "ar_tok_s": result["ar"]["tok_s"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
