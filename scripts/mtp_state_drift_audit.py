#!/usr/bin/env python3
"""Audit MTP verifier commits against a serial AR resident target state.

This is a correctness-only diagnostic for the 35B MTP D64 state-drift lane.  It
drives one persistent-device MTP chain session and one serial AR control session
through the same committed tokens, then compares resident linear-attention state
and live full-attention K/V prefixes after selected MTP cycles.
"""

from __future__ import annotations

import argparse
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

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MTP_CHAIN_CANDIDATE_BUDGETS, MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from hipengine.speculative.mtp_native import NativeMtpChainProposer
from scripts.mtp_prompt_suite_economics import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    PROMPT_RENDER_MODES,
    _load_prompt_encoder,
    _load_prompt_suite,
    _select_prompts,
)


def _parse_csv_ints(value: str | None) -> list[int]:
    if value is None:
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _target_batch(
    root: int,
    context: int,
    candidates: Sequence[int],
    active_count: int,
    *,
    candidate_budget: int,
) -> TargetVerifyBatch:
    return TargetVerifyBatch.from_draft(
        compile_mtp_chain(
            [
                MtpDraftRequest(
                    request_id=0,
                    root_position=int(context),
                    candidate_tokens=tuple(int(token) for token in candidates),
                    active_count=int(active_count),
                )
            ],
            candidate_budget=int(candidate_budget),
        ),
        root_tokens=(int(root),),
        root_positions=(int(context),),
    )


def _capture_tensor(buffer: DeviceBuffer, rows: int, hidden: int) -> Tensor:
    return Tensor.from_handle(buffer.ptr, (int(rows), int(hidden)), DType.BF16, Device("hip", 0))


def _host_dtype(dtype: DType) -> np.dtype[Any] | type[np.generic]:
    if dtype in {DType.BF16, DType.FP16}:
        return np.uint16
    if dtype == DType.FP32:
        return np.float32
    if dtype == DType.INT64:
        return np.int64
    if dtype == DType.INT32:
        return np.int32
    if dtype == DType.BOOL:
        return np.uint8
    if dtype == DType.INT8:
        return np.int8
    raise ValueError(f"unsupported diagnostic tensor dtype: {dtype}")


def _copy_tensor_host(session: Qwen35ParoResidentSession, tensor: Tensor, *, shape: Sequence[int] | None = None) -> np.ndarray:
    out_shape = tuple(int(dim) for dim in (shape if shape is not None else tensor.shape))
    host = np.empty(out_shape, dtype=_host_dtype(tensor.dtype))
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(tensor.ptr), int(host.nbytes)),
        host.nbytes,
        runtime=session.runtime,
    )
    return host


def _copy_tensor_slice_host(
    session: Qwen35ParoResidentSession,
    *,
    ptr: int,
    dtype: DType,
    shape: Sequence[int],
) -> np.ndarray:
    host = np.empty(tuple(int(dim) for dim in shape), dtype=_host_dtype(dtype))
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(ptr), int(host.nbytes)),
        host.nbytes,
        runtime=session.runtime,
    )
    return host


def _first_mismatch(left: np.ndarray, right: np.ndarray) -> dict[str, Any] | None:
    if left.shape != right.shape:
        return {
            "kind": "shape",
            "left_shape": [int(dim) for dim in left.shape],
            "right_shape": [int(dim) for dim in right.shape],
        }
    mismatch = np.not_equal(left, right)
    count = int(np.count_nonzero(mismatch))
    if count == 0:
        return None
    flat = int(np.flatnonzero(mismatch.reshape(-1))[0])
    index = tuple(int(i) for i in np.unravel_index(flat, left.shape))
    record: dict[str, Any] = {
        "kind": "value",
        "mismatch_count": count,
        "first_flat_index": flat,
        "first_index": list(index),
        "left_value": _json_scalar(left[index]),
        "right_value": _json_scalar(right[index]),
    }
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
        diff = np.abs(left.astype(np.float32) - right.astype(np.float32))
        max_flat = int(np.argmax(diff.reshape(-1)))
        max_index = tuple(int(i) for i in np.unravel_index(max_flat, diff.shape))
        record["max_abs"] = float(diff[max_index])
        record["max_abs_index"] = list(max_index)
    return record


def _json_scalar(value: Any) -> int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    raise TypeError(f"unsupported JSON scalar: {type(value)!r}")


def _compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    mismatch = _first_mismatch(left, right)
    result: dict[str, Any] = {
        "passed": mismatch is None,
        "shape": [int(dim) for dim in left.shape],
        "dtype": str(left.dtype),
    }
    if mismatch is not None:
        result["mismatch"] = mismatch
    return result


def _state_compare_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = next((record for record in records if not bool(record.get("passed", False))), None)
    return {
        "passed": first is None,
        "checked": len(records),
        "failed": sum(1 for record in records if not bool(record.get("passed", False))),
        "first_mismatch": first,
    }


def _compare_linear_resident_states(
    left: Qwen35ParoResidentSession,
    right: Qwen35ParoResidentSession,
    *,
    left_label: str,
    right_label: str,
    slot: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in left.linear_layer_ids:
        left_conv, left_recurrent = left._slot_linear_state(int(layer_id), int(slot))
        right_conv, right_recurrent = right._slot_linear_state(int(layer_id), int(slot))
        for state_name, left_tensor, right_tensor in (
            ("conv", left_conv, right_conv),
            ("recurrent", left_recurrent, right_recurrent),
        ):
            left_host = _copy_tensor_host(left, left_tensor)
            right_host = _copy_tensor_host(right, right_tensor)
            record = _compare_arrays(left_host, right_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": state_name,
                    "left": left_label,
                    "right": right_label,
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _compare_mtp_scratch_to_resident(
    session: Qwen35ParoResidentSession,
    *,
    selected_row: int,
    slot: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in session.linear_layer_ids:
        scratch = session.linear_scratch.get(int(layer_id))
        if scratch is None:
            records.append(
                {
                    "passed": False,
                    "layer_index": int(layer_id),
                    "state": "scratch",
                    "mismatch": {"kind": "missing_scratch"},
                }
            )
            continue
        resident_conv, resident_recurrent = session._slot_linear_state(int(layer_id), int(slot))
        for state_name, scratch_tensor, resident_tensor in (
            ("conv", scratch.tree_conv_state, resident_conv),
            ("recurrent", scratch.tree_recurrent_state, resident_recurrent),
        ):
            row_nbytes = int(np.prod(resident_tensor.shape)) * resident_tensor.dtype.itemsize
            scratch_host = _copy_tensor_slice_host(
                session,
                ptr=int(scratch_tensor.ptr) + int(selected_row) * row_nbytes,
                dtype=resident_tensor.dtype,
                shape=resident_tensor.shape,
            )
            resident_host = _copy_tensor_host(session, resident_tensor)
            record = _compare_arrays(scratch_host, resident_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": state_name,
                    "left": f"mtp_scratch_row_{int(selected_row)}",
                    "right": "mtp_resident_slot",
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _compare_mtp_selected_linear_scratch_to_ar(
    mtp: Qwen35ParoResidentSession,
    control: Qwen35ParoResidentSession,
    *,
    selected_row: int,
    slot: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in mtp.linear_layer_ids:
        scratch = mtp.linear_scratch.get(int(layer_id))
        if scratch is None:
            records.append(
                {
                    "passed": False,
                    "layer_index": int(layer_id),
                    "state": "scratch",
                    "mismatch": {"kind": "missing_scratch"},
                }
            )
            continue
        control_conv, control_recurrent = control._slot_linear_state(int(layer_id), int(slot))
        for state_name, scratch_tensor, control_tensor in (
            ("conv", scratch.tree_conv_state, control_conv),
            ("recurrent", scratch.tree_recurrent_state, control_recurrent),
        ):
            row_nbytes = int(np.prod(control_tensor.shape)) * control_tensor.dtype.itemsize
            scratch_host = _copy_tensor_slice_host(
                mtp,
                ptr=int(scratch_tensor.ptr) + int(selected_row) * row_nbytes,
                dtype=control_tensor.dtype,
                shape=control_tensor.shape,
            )
            control_host = _copy_tensor_host(control, control_tensor)
            record = _compare_arrays(scratch_host, control_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": state_name,
                    "left": f"mtp_scratch_row_{int(selected_row)}",
                    "right": "ar_resident",
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _copy_kv_prefix(
    session: Qwen35ParoResidentSession,
    *,
    layer_id: int,
    slot: int,
    live_count: int,
    which: str,
) -> np.ndarray:
    key_cache, value_cache = session._slot_full_cache(int(layer_id), int(slot))
    tensor = key_cache if which == "key" else value_cache
    if tensor.dtype not in {DType.BF16, DType.FP16}:
        raise ValueError(f"KV diagnostic expects 16-bit cache, got {tensor.dtype}")
    if len(tensor.shape) != 4:
        raise ValueError(f"expected rank-4 KV cache, got {tensor.shape}")
    blocks, block_size, heads, head_dim = (int(dim) for dim in tensor.shape)
    if int(live_count) > blocks * block_size:
        raise ValueError("live_count exceeds KV cache capacity")
    shape = (int(live_count), heads, head_dim)
    return _copy_tensor_slice_host(session, ptr=int(tensor.ptr), dtype=tensor.dtype, shape=shape)


def _copy_kv_cell(
    session: Qwen35ParoResidentSession,
    *,
    layer_id: int,
    slot: int,
    position: int,
    which: str,
) -> np.ndarray:
    key_cache, value_cache = session._slot_full_cache(int(layer_id), int(slot))
    tensor = key_cache if which == "key" else value_cache
    if tensor.dtype not in {DType.BF16, DType.FP16}:
        raise ValueError(f"KV diagnostic expects 16-bit cache, got {tensor.dtype}")
    if len(tensor.shape) != 4:
        raise ValueError(f"expected rank-4 KV cache, got {tensor.shape}")
    blocks, block_size, heads, head_dim = (int(dim) for dim in tensor.shape)
    if int(position) < 0 or int(position) >= blocks * block_size:
        raise ValueError("position exceeds KV cache capacity")
    row_elems = heads * head_dim
    ptr = int(tensor.ptr) + int(position) * row_elems * tensor.dtype.itemsize
    return _copy_tensor_slice_host(session, ptr=ptr, dtype=tensor.dtype, shape=(heads, head_dim))


def _compare_full_kv_prefixes(
    left: Qwen35ParoResidentSession,
    right: Qwen35ParoResidentSession,
    *,
    live_count: int,
    left_label: str,
    right_label: str,
    slot: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in left.full_caches:
        for which in ("key", "value"):
            left_host = _copy_kv_prefix(left, layer_id=int(layer_id), slot=int(slot), live_count=int(live_count), which=which)
            right_host = _copy_kv_prefix(right, layer_id=int(layer_id), slot=int(slot), live_count=int(live_count), which=which)
            record = _compare_arrays(left_host, right_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": f"{which}_prefix",
                    "live_count": int(live_count),
                    "left": left_label,
                    "right": right_label,
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _compare_mtp_selected_full_kv_to_ar(
    mtp: Qwen35ParoResidentSession,
    control: Qwen35ParoResidentSession,
    *,
    commit_position: int,
    slot: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for layer_id in mtp.full_caches:
        for which in ("key", "value"):
            mtp_host = _copy_kv_cell(
                mtp,
                layer_id=int(layer_id),
                slot=int(slot),
                position=int(commit_position),
                which=which,
            )
            control_host = _copy_kv_cell(
                control,
                layer_id=int(layer_id),
                slot=int(slot),
                position=int(commit_position),
                which=which,
            )
            record = _compare_arrays(mtp_host, control_host)
            record.update(
                {
                    "layer_index": int(layer_id),
                    "state": f"{which}_cell",
                    "position": int(commit_position),
                    "left": "mtp_selected_kv_cell",
                    "right": "ar_resident_kv_cell",
                }
            )
            records.append(record)
    return _state_compare_summary(records)


def _compare_resident_state(
    mtp: Qwen35ParoResidentSession,
    control: Qwen35ParoResidentSession,
    *,
    live_count: int,
) -> dict[str, Any]:
    linear = _compare_linear_resident_states(mtp, control, left_label="mtp_resident", right_label="ar_resident")
    kv = _compare_full_kv_prefixes(
        mtp,
        control,
        live_count=int(live_count),
        left_label="mtp_resident",
        right_label="ar_resident",
    )
    first = None
    for label, summary in (("linear", linear), ("full_kv_prefix", kv)):
        mismatch = summary.get("first_mismatch")
        if isinstance(mismatch, dict):
            first = {"category": label, **mismatch}
            break
    return {
        "passed": first is None,
        "linear": linear,
        "full_kv_prefix": kv,
        "first_mismatch": first,
    }


def _compare_mtp_selected_state_to_ar(
    mtp: Qwen35ParoResidentSession,
    control: Qwen35ParoResidentSession,
    *,
    selected_row: int,
    commit_position: int,
) -> dict[str, Any]:
    linear = _compare_mtp_selected_linear_scratch_to_ar(
        mtp,
        control,
        selected_row=int(selected_row),
    )
    kv = _compare_mtp_selected_full_kv_to_ar(
        mtp,
        control,
        commit_position=int(commit_position),
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


def _resolve_prompt_tokens(args: argparse.Namespace) -> tuple[str, list[int], dict[str, Any]]:
    raw_tokens = str(getattr(args, "prompt_tokens", "") or "").strip()
    if raw_tokens:
        tokens = _parse_csv_ints(raw_tokens)
        if not tokens:
            raise ValueError("--prompt-tokens did not contain any token ids")
        return "inline", tokens, {"source": "inline --prompt-tokens"}
    suite = _load_prompt_suite(Path(args.prompts))
    selected = _select_prompts(suite, names_csv=str(args.prompt_name), limit=1)
    if len(selected) != 1:
        raise ValueError("--prompt-name must select exactly one prompt")
    encoder = _load_prompt_encoder(Path(args.model), str(args.prompt_render))
    encoded = encoder.encode(selected[0]["prompt"])
    return (
        str(selected[0]["name"]),
        [int(token) for token in encoded.token_ids],
        {
            "source": str(args.prompts),
            "prompt_render": str(args.prompt_render),
            "tokenization": encoder.tokenization,
            "source_text": selected[0]["prompt"],
            "rendered_text": encoded.rendered_text,
        },
    )


def _prefill_serial(
    session: Qwen35ParoResidentSession,
    prompt_tokens: Sequence[int],
    *,
    capture: Tensor | None = None,
    capture_layer_id: int | None = None,
) -> int:
    next_result = None
    for pos, token in enumerate(prompt_tokens):
        if capture is None:
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_tokens) - 1))
        else:
            next_result = session.step_with_hidden_taps(
                int(token),
                position=pos,
                capture_layer_ids=(int(capture_layer_id),),
                capture_hidden_concat=capture,
                capture_row=pos,
                sample=(pos == len(prompt_tokens) - 1),
            )
    if next_result is None:
        raise RuntimeError("prompt did not produce a root token")
    return int(next_result.token_id)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.candidate_budget) not in MTP_CHAIN_CANDIDATE_BUDGETS:
        raise ValueError(f"candidate budget must be one of {sorted(MTP_CHAIN_CANDIDATE_BUDGETS)}")
    if str(args.chain_attn_mode) == "decode_batched" and str(args.graph_mode) != "off":
        raise ValueError("decode_batched currently requires graph-mode=off")
    prompt_name, prompt_tokens, prompt_meta = _resolve_prompt_tokens(args)
    compare_cycles = set(_parse_csv_ints(args.compare_after_cycles))
    if not compare_cycles:
        compare_cycles = set(range(1, int(args.max_cycles) + 1))
    max_cycle = max(compare_cycles)
    max_cycles = min(int(args.max_cycles), max_cycle)
    if max_cycles <= 0:
        raise ValueError("max cycles must be positive")

    runner = Qwen35ParoNextTokenRunner(Path(args.model), backend=str(args.backend))
    candidate_budget = int(args.candidate_budget)
    max_sequence = len(prompt_tokens) + int(args.decode_tokens) + candidate_budget + 4
    capture_rows = max_sequence + 2
    started = time.perf_counter()
    generated: list[int] = []
    cycle_records: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    capture_buf: DeviceBuffer | None = None
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_batch_size=candidate_budget + 1,
    ) as mtp_session, Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_batch_size=1,
    ) as control_session:
        hidden = int(mtp_session.config.hidden_size)
        capture_layer_id = int(mtp_session.layer_limit) - 1
        capture_buf = malloc(capture_rows * hidden * DType.BF16.itemsize, runtime=mtp_session.runtime)
        try:
            capture = _capture_tensor(capture_buf, capture_rows, hidden)
            verifier_no_capture = Tensor.from_handle(0, (candidate_budget + 1, 0), DType.BF16, Device("hip", 0))
            mtp_root = _prefill_serial(
                mtp_session,
                prompt_tokens,
                capture=capture,
                capture_layer_id=capture_layer_id,
            )
            control_root = _prefill_serial(control_session, prompt_tokens)
            context = len(prompt_tokens)
            control_context = len(prompt_tokens)
            prefill_compare = _compare_resident_state(mtp_session, control_session, live_count=context)
            if not bool(prefill_compare["passed"]):
                first_mismatch = {"cycle": 0, "phase": "prefill", **prefill_compare["first_mismatch"]}
            with NativeMtpChainProposer(
                Path(args.model),
                max_positions=max_sequence + int(args.decode_tokens) + 4,
                max_mtp_tokens=len(prompt_tokens) + 2 * int(args.decode_tokens) + 8,
                runtime=mtp_session.runtime,
            ) as proposer:
                proposer.prefill_from_target_hidden_rows(
                    prompt_tokens,
                    capture_base_ptr=capture_buf.ptr,
                    seed_token=mtp_root,
                    read_expert_topk=False,
                    read_lm_head_value=False,
                )
                for cycle in range(1, max_cycles + 1):
                    remaining = int(args.decode_tokens) - len(generated)
                    active_budget = min(candidate_budget, max(0, remaining - 1))
                    if active_budget <= 0:
                        break
                    snapshots = [proposer.save_state(0)]
                    candidates = [int(proposer.current.token)]
                    for draft_idx in range(1, active_budget):
                        proposer.advance_with_previous_hidden(
                            input_token=candidates[-1],
                            position=proposer.position + 1,
                            read_expert_topk=False,
                            read_lm_head_value=False,
                        )
                        if draft_idx < active_budget - 1:
                            snapshots.append(proposer.save_state(draft_idx))
                        candidates.append(int(proposer.current.token))
                    active_budget = len(candidates)
                    target_batch = _target_batch(
                        mtp_root,
                        context,
                        candidates,
                        active_budget,
                        candidate_budget=candidate_budget,
                    )
                    verify = mtp_session.verify_chain_bulk_and_commit(
                        target_batch,
                        base_slot=0,
                        capture_layer_ids=(),
                        capture_hidden_concat=verifier_no_capture,
                        capture_row_start=0,
                        chain_attn_mode=str(args.chain_attn_mode),
                        graph_mode=str(args.graph_mode),
                        canonicalize_after=not _env_truthy("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", default=True),
                    )
                    accepted = int(verify.accepted_count)
                    accepted_tokens = candidates[:accepted]
                    committed = [int(mtp_root), *accepted_tokens]
                    bonus = int(verify.next_token) if verify.next_token is not None else int(verify.target_top1[min(accepted, len(verify.target_top1) - 1)])
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
                    compare_payload = None
                    scratch_payload = None
                    selected_payload = None
                    if cycle in compare_cycles:
                        compare_payload = _compare_resident_state(mtp_session, control_session, live_count=context)
                        scratch_payload = _compare_mtp_scratch_to_resident(
                            mtp_session,
                            selected_row=int(verify.commit_row),
                        )
                        selected_payload = _compare_mtp_selected_state_to_ar(
                            mtp_session,
                            control_session,
                            selected_row=int(verify.commit_row),
                            commit_position=int(verify.commit_position),
                        )
                        if first_mismatch is None and not bool(compare_payload["passed"]):
                            first_mismatch = {
                                "cycle": int(cycle),
                                "phase": "post_commit",
                                **compare_payload["first_mismatch"],
                            }
                    record: dict[str, Any] = {
                        "cycle": int(cycle),
                        "context_before": int(context - len(committed)),
                        "context_after": int(context),
                        "committed_tokens": committed,
                        "draft_candidates": [int(token) for token in candidates],
                        "accepted": accepted,
                        "commit_row": int(verify.commit_row),
                        "commit_position": int(verify.commit_position),
                        "next_token": int(bonus),
                        "control_next_token": int(control_bonus),
                        "control_next_token_match": int(bonus) == int(control_bonus),
                        "control_token_mismatches": control_token_mismatches,
                        "graph": verify.graph,
                    }
                    if compare_payload is not None:
                        record["resident_vs_ar"] = compare_payload
                    if scratch_payload is not None:
                        record["mtp_scratch_vs_resident"] = scratch_payload
                    if selected_payload is not None:
                        record["mtp_selected_state_vs_ar"] = selected_payload
                    cycle_records.append(record)
                    if len(generated) >= int(args.decode_tokens):
                        break
                    if accepted < active_budget - 1:
                        proposer.restore_state(snapshots[accepted])
                    elif accepted >= active_budget:
                        proposer.advance_with_previous_hidden(
                            input_token=candidates[-1],
                            position=proposer.position + 1,
                            need_result=False,
                            read_expert_topk=False,
                            read_lm_head_value=False,
                        )
                    proposer.advance_with_previous_hidden(
                        input_token=bonus,
                        position=proposer.position + 1,
                        read_expert_topk=False,
                        read_lm_head_value=False,
                    )
                    mtp_root = bonus
        finally:
            if capture_buf is not None:
                free(capture_buf, runtime=mtp_session.runtime)
    seconds = time.perf_counter() - started
    return {
        "status": "passed" if first_mismatch is None else "state_mismatch",
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
            "compare_after_cycles": sorted(int(cycle) for cycle in compare_cycles),
            "max_cycles": int(max_cycles),
        },
        "prompt": prompt_meta,
        "seconds": float(seconds),
        "generated_prefix": generated[: int(args.decode_tokens)],
        "first_mismatch": first_mismatch,
        "cycles": cycle_records,
        "note": "Correctness diagnostic only: compares resident MTP target state with a serial AR control after committed tokens.",
    }


def _env_truthy(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    parser.add_argument("--compare-after-cycles", default="1,2,3,4", help="comma-separated MTP cycle numbers to compare")
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload["status"] in {"passed", "state_mismatch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
