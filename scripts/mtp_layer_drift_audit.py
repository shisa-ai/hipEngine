#!/usr/bin/env python3
"""Compare drifted MTP verifier layers against a clean AR verifier state.

This is a correctness-only diagnostic for the D64 `translation` MTP drift lane.
It replays persistent-device MTP up to a selected cycle while keeping a serial
AR control session at the same committed prefix.  At the selected cycle it runs
the same root+candidate verifier batch on both sessions, captures BF16 hidden
rows after each layer, and reports where hidden/logit divergence appears.
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
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MTP_CHAIN_CANDIDATE_BUDGETS
from hipengine.speculative.mtp_native import NativeMtpChainProposer
from scripts.mtp_prompt_suite_economics import DEFAULT_MODEL, DEFAULT_PROMPTS, PROMPT_RENDER_MODES
from scripts.mtp_state_drift_audit import (
    _compare_arrays,
    _compare_resident_state,
    _copy_kv_cell,
    _copy_tensor_slice_host,
    _env_truthy,
    _parse_csv_ints,
    _prefill_serial,
    _resolve_prompt_tokens,
    _state_compare_summary,
    _target_batch,
)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _capture_tensor(buffer: DeviceBuffer, rows: int, width: int) -> Tensor:
    return Tensor.from_handle(buffer.ptr, (int(rows), int(width)), DType.BF16, Device("hip", 0))


def _capture_layer_row(
    session: Qwen35ParoResidentSession,
    capture: Tensor,
    *,
    row: int,
    capture_index: int,
    hidden_size: int,
) -> np.ndarray:
    width = int(capture.shape[1])
    offset_elems = int(row) * width + int(capture_index) * int(hidden_size)
    return _copy_tensor_slice_host(
        session,
        ptr=int(capture.ptr) + offset_elems * DType.BF16.itemsize,
        dtype=DType.BF16,
        shape=(int(hidden_size),),
    )


def _compare_bf16_vectors(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "passed": False,
            "kind": "shape",
            "left_shape": [int(dim) for dim in left.shape],
            "right_shape": [int(dim) for dim in right.shape],
        }
    mismatch = np.not_equal(left, right)
    count = int(np.count_nonzero(mismatch))
    left_f32 = _bf16_bits_to_f32(left)
    right_f32 = _bf16_bits_to_f32(right)
    diff = np.abs(left_f32 - right_f32)
    max_flat = int(np.nanargmax(diff.reshape(-1))) if diff.size else 0
    max_abs = float(diff.reshape(-1)[max_flat]) if diff.size else 0.0
    result: dict[str, Any] = {
        "passed": count == 0,
        "mismatch_count": count,
        "max_abs": max_abs,
        "max_abs_index": max_flat,
        "left_at_max_abs": float(left_f32.reshape(-1)[max_flat]) if diff.size else 0.0,
        "right_at_max_abs": float(right_f32.reshape(-1)[max_flat]) if diff.size else 0.0,
        "left_bits_at_max_abs": int(left.reshape(-1)[max_flat]) if diff.size else 0,
        "right_bits_at_max_abs": int(right.reshape(-1)[max_flat]) if diff.size else 0,
    }
    if count:
        first_flat = int(np.flatnonzero(mismatch.reshape(-1))[0])
        result.update(
            {
                "first_mismatch_index": first_flat,
                "left_first_bits": int(left.reshape(-1)[first_flat]),
                "right_first_bits": int(right.reshape(-1)[first_flat]),
                "left_first_value": float(left_f32.reshape(-1)[first_flat]),
                "right_first_value": float(right_f32.reshape(-1)[first_flat]),
            }
        )
    return result


def _compare_hidden_captures(
    mtp: Qwen35ParoResidentSession,
    clean: Qwen35ParoResidentSession,
    *,
    mtp_capture: Tensor,
    clean_capture: Tensor,
    capture_layer_ids: Sequence[int],
    rows: Sequence[int],
) -> dict[str, Any]:
    hidden_size = int(mtp.config.hidden_size)
    records: list[dict[str, Any]] = []
    for capture_index, layer_id in enumerate(capture_layer_ids):
        layer_type = str(mtp.config.layer_types[int(layer_id)])
        for row in rows:
            left = _capture_layer_row(
                mtp,
                mtp_capture,
                row=int(row),
                capture_index=int(capture_index),
                hidden_size=hidden_size,
            )
            right = _capture_layer_row(
                clean,
                clean_capture,
                row=int(row),
                capture_index=int(capture_index),
                hidden_size=hidden_size,
            )
            record = _compare_bf16_vectors(left, right)
            record.update(
                {
                    "row": int(row),
                    "layer_index": int(layer_id),
                    "layer_type": layer_type,
                    "capture_index": int(capture_index),
                }
            )
            records.append(record)
    first_bit_mismatch = next((record for record in records if int(record.get("mismatch_count", 0)) > 0), None)
    top_max_abs = sorted(records, key=lambda item: float(item.get("max_abs", 0.0)), reverse=True)[:10]
    return {
        "passed": first_bit_mismatch is None,
        "checked": len(records),
        "failed": sum(1 for record in records if int(record.get("mismatch_count", 0)) > 0),
        "first_bit_mismatch": first_bit_mismatch,
        "top_max_abs": top_max_abs,
        "records": records,
    }


def _copy_scratch_row(
    session: Qwen35ParoResidentSession,
    tensor: Tensor,
    *,
    row: int,
) -> np.ndarray:
    if int(tensor.shape[0]) <= int(row):
        raise ValueError(f"scratch row {row} outside tensor shape {tensor.shape}")
    row_shape = tuple(int(dim) for dim in tensor.shape[1:])
    row_nbytes = int(np.prod(row_shape)) * tensor.dtype.itemsize
    return _copy_tensor_slice_host(
        session,
        ptr=int(tensor.ptr) + int(row) * row_nbytes,
        dtype=tensor.dtype,
        shape=row_shape,
    )


def _compare_selected_linear_scratch(
    left: Qwen35ParoResidentSession,
    right: Qwen35ParoResidentSession,
    *,
    left_row: int,
    right_row: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in left.linear_layer_ids:
        left_scratch = left.linear_scratch.get(int(layer_id))
        right_scratch = right.linear_scratch.get(int(layer_id))
        if left_scratch is None or right_scratch is None:
            records.append(
                {
                    "passed": False,
                    "layer_index": int(layer_id),
                    "state": "scratch",
                    "mismatch": {
                        "kind": "missing_scratch",
                        "left_missing": left_scratch is None,
                        "right_missing": right_scratch is None,
                    },
                }
            )
            continue
        for state_name, left_tensor, right_tensor in (
            ("conv", left_scratch.tree_conv_state, right_scratch.tree_conv_state),
            ("recurrent", left_scratch.tree_recurrent_state, right_scratch.tree_recurrent_state),
        ):
            left_host = _copy_scratch_row(left, left_tensor, row=int(left_row))
            right_host = _copy_scratch_row(right, right_tensor, row=int(right_row))
            record = _compare_arrays(left_host, right_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": state_name,
                    "left": f"drifted_scratch_row_{int(left_row)}",
                    "right": f"clean_scratch_row_{int(right_row)}",
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _compare_selected_full_kv_cells(
    left: Qwen35ParoResidentSession,
    right: Qwen35ParoResidentSession,
    *,
    left_position: int,
    right_position: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in left.full_caches:
        for which in ("key", "value"):
            left_host = _copy_kv_cell(
                left,
                layer_id=int(layer_id),
                slot=0,
                position=int(left_position),
                which=which,
            )
            right_host = _copy_kv_cell(
                right,
                layer_id=int(layer_id),
                slot=0,
                position=int(right_position),
                which=which,
            )
            record = _compare_arrays(left_host, right_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": f"{which}_cell",
                    "left_position": int(left_position),
                    "right_position": int(right_position),
                    "left": "drifted_verify_kv_cell",
                    "right": "clean_verify_kv_cell",
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _compare_selected_verify_state(
    left: Qwen35ParoResidentSession,
    right: Qwen35ParoResidentSession,
    *,
    left_row: int,
    right_row: int,
    left_position: int,
    right_position: int,
) -> dict[str, Any]:
    linear = _compare_selected_linear_scratch(
        left,
        right,
        left_row=int(left_row),
        right_row=int(right_row),
    )
    kv = _compare_selected_full_kv_cells(
        left,
        right,
        left_position=int(left_position),
        right_position=int(right_position),
    )
    first = None
    for label, summary in (("linear_scratch", linear), ("full_kv_cell", kv)):
        mismatch = summary.get("first_mismatch")
        if isinstance(mismatch, dict):
            first = {"category": label, **mismatch}
            break
    return {
        "passed": first is None,
        "linear_scratch": linear,
        "full_kv_cell": kv,
        "first_mismatch": first,
    }


def _build_candidates(
    proposer: NativeMtpChainProposer,
    *,
    active_budget: int,
) -> tuple[list[int], list[Any]]:
    snapshots = [proposer.save_state(0)]
    candidates = [int(proposer.current.token)]
    for draft_idx in range(1, int(active_budget)):
        proposer.advance_with_previous_hidden(
            input_token=candidates[-1],
            position=proposer.position + 1,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        if draft_idx < int(active_budget) - 1:
            snapshots.append(proposer.save_state(draft_idx))
        candidates.append(int(proposer.current.token))
    return candidates, snapshots


def _advance_proposer_after_verify(
    proposer: NativeMtpChainProposer,
    *,
    accepted: int,
    active_budget: int,
    candidates: Sequence[int],
    snapshots: Sequence[Any],
    bonus: int,
) -> None:
    if int(accepted) < int(active_budget) - 1:
        proposer.restore_state(snapshots[int(accepted)])
    elif int(accepted) >= int(active_budget):
        proposer.advance_with_previous_hidden(
            input_token=int(candidates[-1]),
            position=proposer.position + 1,
            need_result=False,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
    proposer.advance_with_previous_hidden(
        input_token=int(bonus),
        position=proposer.position + 1,
        read_expert_topk=False,
        read_lm_head_value=False,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.candidate_budget) not in MTP_CHAIN_CANDIDATE_BUDGETS:
        raise ValueError(f"candidate budget must be one of {sorted(MTP_CHAIN_CANDIDATE_BUDGETS)}")
    if str(args.chain_attn_mode) == "decode_batched" and str(args.graph_mode) != "off":
        raise ValueError("decode_batched currently requires graph-mode=off")
    prompt_name, prompt_tokens, prompt_meta = _resolve_prompt_tokens(args)
    compare_cycle = int(args.compare_cycle)
    if compare_cycle <= 0:
        raise ValueError("--compare-cycle must be positive")
    rows_to_compare = _parse_csv_ints(args.compare_rows)
    if not rows_to_compare:
        rows_to_compare = [0]

    runner = Qwen35ParoNextTokenRunner(Path(args.model), backend=str(args.backend))
    candidate_budget = int(args.candidate_budget)
    max_sequence = len(prompt_tokens) + int(args.decode_tokens) + candidate_budget + 4
    started = time.perf_counter()
    generated: list[int] = []
    cycle_records: list[dict[str, Any]] = []
    prompt_capture_buf: DeviceBuffer | None = None
    mtp_layer_capture_buf: DeviceBuffer | None = None
    clean_layer_capture_buf: DeviceBuffer | None = None
    compare_payload: dict[str, Any] | None = None

    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_batch_size=candidate_budget + 1,
    ) as mtp_session, Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_batch_size=candidate_budget + 1,
    ) as control_session:
        hidden = int(mtp_session.config.hidden_size)
        prompt_capture_rows = max_sequence + 2
        prompt_capture_layer_id = int(mtp_session.layer_limit) - 1
        capture_layer_ids = tuple(range(int(mtp_session.layer_limit)))
        capture_width = len(capture_layer_ids) * hidden
        verify_rows = candidate_budget + 1
        for row in rows_to_compare:
            if int(row) < 0 or int(row) >= verify_rows:
                raise ValueError(f"compare row {row} outside verifier rows {verify_rows}")

        prompt_capture_buf = malloc(prompt_capture_rows * hidden * DType.BF16.itemsize, runtime=mtp_session.runtime)
        mtp_layer_capture_buf = malloc(verify_rows * capture_width * DType.BF16.itemsize, runtime=mtp_session.runtime)
        clean_layer_capture_buf = malloc(verify_rows * capture_width * DType.BF16.itemsize, runtime=control_session.runtime)
        try:
            prompt_capture = _capture_tensor(prompt_capture_buf, prompt_capture_rows, hidden)
            mtp_layer_capture = _capture_tensor(mtp_layer_capture_buf, verify_rows, capture_width)
            clean_layer_capture = _capture_tensor(clean_layer_capture_buf, verify_rows, capture_width)
            verifier_no_capture = Tensor.from_handle(0, (verify_rows, 0), DType.BF16, Device("hip", 0))

            mtp_root = _prefill_serial(
                mtp_session,
                prompt_tokens,
                capture=prompt_capture,
                capture_layer_id=prompt_capture_layer_id,
            )
            control_root = _prefill_serial(control_session, prompt_tokens)
            context = len(prompt_tokens)
            control_context = len(prompt_tokens)

            with NativeMtpChainProposer(
                Path(args.model),
                max_positions=max_sequence + int(args.decode_tokens) + 4,
                max_mtp_tokens=len(prompt_tokens) + 2 * int(args.decode_tokens) + 8,
                runtime=mtp_session.runtime,
            ) as proposer:
                proposer.prefill_from_target_hidden_rows(
                    prompt_tokens,
                    capture_base_ptr=prompt_capture_buf.ptr,
                    seed_token=mtp_root,
                    read_expert_topk=False,
                    read_lm_head_value=False,
                )
                for cycle in range(1, compare_cycle + 1):
                    remaining = int(args.decode_tokens) - len(generated)
                    active_budget = min(candidate_budget, max(0, remaining - 1))
                    if active_budget <= 0:
                        break
                    candidates, snapshots = _build_candidates(proposer, active_budget=active_budget)
                    active_budget = len(candidates)
                    target_batch = _target_batch(
                        mtp_root,
                        context,
                        candidates,
                        active_budget,
                        candidate_budget=candidate_budget,
                    )
                    is_compare_cycle = cycle == compare_cycle
                    pre_compare_state = None
                    if is_compare_cycle:
                        pre_compare_state = _compare_resident_state(mtp_session, control_session, live_count=context)
                    verify = mtp_session.verify_chain_bulk_and_commit(
                        target_batch,
                        base_slot=0,
                        capture_layer_ids=capture_layer_ids if is_compare_cycle else (),
                        capture_hidden_concat=mtp_layer_capture if is_compare_cycle else verifier_no_capture,
                        capture_row_start=0,
                        chain_attn_mode=str(args.chain_attn_mode),
                        graph_mode=str(args.graph_mode),
                        canonicalize_after=not _env_truthy("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", default=True),
                    )
                    mtp_top1, mtp_top1_values = mtp_session._read_verify_top1(int(verify.rows))
                    accepted = int(verify.accepted_count)
                    accepted_tokens = candidates[:accepted]
                    committed = [int(mtp_root), *accepted_tokens]
                    bonus = int(verify.next_token) if verify.next_token is not None else int(mtp_top1[min(accepted, len(mtp_top1) - 1)])

                    if is_compare_cycle:
                        clean_batch = _target_batch(
                            control_root,
                            control_context,
                            candidates,
                            active_budget,
                            candidate_budget=candidate_budget,
                        )
                        clean_verify = control_session.verify_chain_bulk_and_commit(
                            clean_batch,
                            base_slot=0,
                            capture_layer_ids=capture_layer_ids,
                            capture_hidden_concat=clean_layer_capture,
                            capture_row_start=0,
                            chain_attn_mode=str(args.chain_attn_mode),
                            graph_mode=str(args.graph_mode),
                            canonicalize_after=not _env_truthy("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", default=True),
                        )
                        clean_top1, clean_top1_values = control_session._read_verify_top1(int(clean_verify.rows))
                        hidden_compare = _compare_hidden_captures(
                            mtp_session,
                            control_session,
                            mtp_capture=mtp_layer_capture,
                            clean_capture=clean_layer_capture,
                            capture_layer_ids=capture_layer_ids,
                            rows=rows_to_compare,
                        )
                        selected_state_compare = _compare_selected_verify_state(
                            mtp_session,
                            control_session,
                            left_row=int(verify.commit_row),
                            right_row=int(clean_verify.commit_row),
                            left_position=int(verify.commit_position),
                            right_position=int(clean_verify.commit_position),
                        )
                        compare_payload = {
                            "cycle": int(cycle),
                            "context": int(context),
                            "control_context": int(control_context),
                            "root": int(mtp_root),
                            "control_root": int(control_root),
                            "draft_candidates": [int(token) for token in candidates],
                            "pre_verify_resident_vs_ar": pre_compare_state,
                            "mtp_verify": {
                                "accepted": int(verify.accepted_count),
                                "commit_row": int(verify.commit_row),
                                "commit_position": int(verify.commit_position),
                                "next_token": int(bonus),
                                "target_top1": [int(token) for token in mtp_top1],
                                "target_top1_values": [float(value) for value in mtp_top1_values],
                            },
                            "clean_verify": {
                                "accepted": int(clean_verify.accepted_count),
                                "commit_row": int(clean_verify.commit_row),
                                "commit_position": int(clean_verify.commit_position),
                                "next_token": int(clean_verify.next_token) if clean_verify.next_token is not None else int(clean_top1[min(int(clean_verify.accepted_count), len(clean_top1) - 1)]),
                                "target_top1": [int(token) for token in clean_top1],
                                "target_top1_values": [float(value) for value in clean_top1_values],
                            },
                            "hidden_compare": hidden_compare,
                            "selected_state_compare": selected_state_compare,
                        }
                        break

                    control_bonus = control_root
                    control_token_mismatches: list[dict[str, Any]] = []
                    for token in committed:
                        if int(token) != int(control_root):
                            control_token_mismatches.append(
                                {
                                    "position": int(control_context),
                                    "expected_ar_root": int(control_root),
                                    "mtp_committed": int(token),
                                }
                            )
                        step_result = control_session.step(int(token), position=control_context, sample=True)
                        if step_result is None:
                            raise RuntimeError("control AR step produced no token")
                        control_bonus = int(step_result.token_id)
                        control_root = control_bonus
                        control_context += 1
                    generated.extend(committed)
                    context += len(committed)
                    cycle_records.append(
                        {
                            "cycle": int(cycle),
                            "context_before": int(context - len(committed)),
                            "context_after": int(context),
                            "committed_tokens": committed,
                            "draft_candidates": [int(token) for token in candidates],
                            "accepted": accepted,
                            "next_token": int(bonus),
                            "control_next_token": int(control_bonus),
                            "control_next_token_match": int(bonus) == int(control_bonus),
                            "control_token_mismatches": control_token_mismatches,
                        }
                    )
                    if len(generated) >= int(args.decode_tokens):
                        break
                    _advance_proposer_after_verify(
                        proposer,
                        accepted=accepted,
                        active_budget=active_budget,
                        candidates=candidates,
                        snapshots=snapshots,
                        bonus=bonus,
                    )
                    mtp_root = bonus
        finally:
            if prompt_capture_buf is not None:
                free(prompt_capture_buf, runtime=mtp_session.runtime)
            if mtp_layer_capture_buf is not None:
                free(mtp_layer_capture_buf, runtime=mtp_session.runtime)
            if clean_layer_capture_buf is not None:
                free(clean_layer_capture_buf, runtime=control_session.runtime)

    seconds = time.perf_counter() - started
    status = "compared" if compare_payload is not None else "compare_cycle_not_reached"
    return {
        "status": status,
        "performance_claim": False,
        "model": str(args.model),
        "backend": str(args.backend),
        "workload": {
            "prompt_name": prompt_name,
            "prompt_tokens": len(prompt_tokens),
            "decode_tokens": int(args.decode_tokens),
            "candidate_budget": candidate_budget,
            "chain_attn_mode": str(args.chain_attn_mode),
            "graph_mode": str(args.graph_mode),
            "compare_cycle": compare_cycle,
            "compare_rows": [int(row) for row in rows_to_compare],
        },
        "prompt": prompt_meta,
        "seconds": float(seconds),
        "generated_prefix_before_compare": generated[: int(args.decode_tokens)],
        "cycles_before_compare": cycle_records,
        "comparison": compare_payload,
        "note": "Correctness diagnostic only: compares verifier hidden taps from drifted MTP state against a clean AR-state verifier.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-name", default="translation")
    parser.add_argument("--prompt-render", choices=PROMPT_RENDER_MODES, default="raw")
    parser.add_argument("--prompt-tokens", default="", help="comma-separated token ids; bypasses --prompt-name")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--candidate-budget", type=int, default=1)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="decode_batched")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--compare-cycle", type=int, default=27)
    parser.add_argument("--compare-rows", default="0,1", help="comma-separated verifier rows to compare")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload["status"] in {"compared", "compare_cycle_not_reached"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
