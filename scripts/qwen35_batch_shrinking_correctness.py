#!/usr/bin/env python3
"""Qwen3.5/PARO c=N to c=1 lifecycle-shape correctness gate.

The gate starts with a packed prompt batch, decodes a fixed number of tokens at
each live width, and retires rows in a hole-producing order. Prompts may be
ragged, one row may finish through the scheduler's EOS event, and a configurable
physical slot survives to c=1. Generated token IDs plus every linear recurrent
state and full-attention KV prefix are compared with independent c=1 runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.generation import GeneratedToken, PerRowSamplingParams, ResidentBatchScheduler
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_retained_bench import (
    DEFAULT_FIXTURE,
    DEFAULT_MODEL,
    _compiler_version,
    _hardware_context,
    _isolated_c1_route_env,
    _load_prompt_slices,
    _software_context,
)


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_shrinking_correctness.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _parse_prompt_lengths(
    value: str | None,
    *,
    batch_size: int,
    fallback_length: int,
) -> tuple[int, ...]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if value is None:
        lengths = (int(fallback_length),) * int(batch_size)
    else:
        try:
            lengths = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
        except ValueError as exc:
            raise ValueError("prompt_lengths must be comma-separated integers") from exc
        if len(lengths) != int(batch_size):
            raise ValueError("prompt_lengths must contain exactly batch_size entries")
    if any(length <= 0 for length in lengths):
        raise ValueError("prompt_lengths entries must be positive")
    return lengths


def _load_ragged_prompt_slices(path: Path, prompt_lengths: Sequence[int]) -> list[list[int]]:
    lengths = tuple(int(length) for length in prompt_lengths)
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("prompt_lengths entries must be positive")
    rows = _load_prompt_slices(
        path,
        prompt_length=max(lengths),
        batch_size=len(lengths),
    )
    return [row[:length] for row, length in zip(rows, lengths, strict=True)]


def _cancellation_order(batch_size: int, *, survivor_slot: int = 0) -> tuple[int, ...]:
    """Create holes early while leaving ``survivor_slot`` resident to c=1."""

    rows = int(batch_size)
    survivor = int(survivor_slot)
    if rows < 2:
        raise ValueError("batch_size must be at least two")
    if survivor < 0 or survivor >= rows:
        raise ValueError("survivor_slot must be within batch_size")

    pending = [slot for slot in range(rows) if slot != survivor]
    order: list[int] = []
    while pending:
        index = len(pending) // 2
        order.append(pending.pop(index))
    return tuple(order)


def _decode_counts_for_order(
    batch_size: int,
    order: Sequence[int],
    *,
    steps_per_width: int,
) -> list[int]:
    rows = int(batch_size)
    steps = int(steps_per_width)
    if rows < 2:
        raise ValueError("batch_size must be at least two")
    if steps <= 0:
        raise ValueError("steps_per_width must be positive")
    termination_order = tuple(int(slot) for slot in order)
    if len(termination_order) != rows - 1 or len(set(termination_order)) != rows - 1:
        raise ValueError("termination order must contain batch_size - 1 unique slots")
    if any(slot < 0 or slot >= rows for slot in termination_order):
        raise ValueError("termination order slots must be within batch_size")
    survivor = next(slot for slot in range(rows) if slot not in termination_order)
    counts = [0] * rows
    for stage_index, slot in enumerate(termination_order):
        counts[slot] = (stage_index + 1) * steps
    counts[survivor] = rows * steps
    return counts


def _device_sha256(session: Qwen35ParoResidentSession, ptr: int, nbytes: int) -> str:
    size = int(nbytes)
    if size <= 0:
        raise ValueError("device snapshot size must be positive")
    host = np.empty((size,), dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(ptr), size),
        runtime=session.runtime,
    )
    return hashlib.sha256(host.tobytes()).hexdigest()


def _slot_state_snapshot(
    session: Qwen35ParoResidentSession,
    *,
    slot: int,
    live_count: int,
) -> dict[str, Any]:
    """Hash all persistent model state for one physical request slot."""

    if int(live_count) <= 0:
        raise ValueError("live_count must be positive")
    session.runtime.device_synchronize()
    linear: dict[str, Any] = {}
    full_kv: dict[str, Any] = {}
    layer_types = tuple(str(layer_type) for layer_type in getattr(session.config, "layer_types", ()))
    for layer_id, layer_type in enumerate(layer_types[: len(session.states)]):
        if layer_type == "linear_attention":
            conv_state, recurrent_state = session._slot_linear_state(layer_id, int(slot))
            linear[str(layer_id)] = {
                "conv_sha256": _device_sha256(
                    session,
                    conv_state.ptr,
                    conv_state.numel * conv_state.dtype.itemsize,
                ),
                "recurrent_sha256": _device_sha256(
                    session,
                    recurrent_state.ptr,
                    recurrent_state.numel * recurrent_state.dtype.itemsize,
                ),
            }
        elif layer_type == "full_attention":
            key_cache, value_cache = session._slot_full_cache(layer_id, int(slot))
            blocks, block_size, num_heads, head_dim = (int(dim) for dim in key_cache.shape)
            if int(live_count) > blocks * block_size:
                raise ValueError("live_count exceeds full-attention KV capacity")
            prefix_nbytes = int(live_count) * num_heads * head_dim * key_cache.dtype.itemsize
            full_kv[str(layer_id)] = {
                "key_prefix_sha256": _device_sha256(session, key_cache.ptr, prefix_nbytes),
                "value_prefix_sha256": _device_sha256(session, value_cache.ptr, prefix_nbytes),
            }
    snapshot: dict[str, Any] = {
        "live_count": int(live_count),
        "linear": linear,
        "full_kv": full_kv,
    }
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["aggregate_sha256"] = hashlib.sha256(canonical).hexdigest()
    return snapshot


def _state_snapshot_mismatches(batch: dict[str, Any], c1: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if batch.get("live_count") != c1.get("live_count"):
        mismatches.append("live_count")
    for family in ("linear", "full_kv"):
        batch_family = batch.get(family, {})
        c1_family = c1.get(family, {})
        for layer_id in sorted(set(batch_family) | set(c1_family), key=int):
            batch_layer = batch_family.get(layer_id, {})
            c1_layer = c1_family.get(layer_id, {})
            for field in sorted(set(batch_layer) | set(c1_layer)):
                if batch_layer.get(field) != c1_layer.get(field):
                    mismatches.append(f"{family}.{layer_id}.{field}")
    return mismatches


def _run_c1_reference(
    runner: Qwen35ParoNextTokenRunner,
    prompts: Sequence[Sequence[int]],
    decode_counts: Sequence[int],
    *,
    max_layers: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    rows: list[list[int]] = []
    state_snapshots: list[dict[str, Any]] = []
    with (
        _isolated_c1_route_env(),
        Qwen35ParoResidentSession(
            runner,
            max_sequence_length=max_sequence_length,
            max_layers=max_layers,
            max_batch_size=1,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
        ) as session,
    ):
        for prompt, decode_tokens in zip(prompts, decode_counts, strict=True):
            seed = session.prefill_native(prompt, sample=True)
            if seed is None:
                raise RuntimeError("c1 prefill did not produce a seed token")
            next_token = int(seed.token_id)
            sequence = [next_token]
            for step in range(int(decode_tokens)):
                result = session.step(next_token, position=len(prompt) + step, sample=True)
                if result is None:
                    raise RuntimeError("c1 decode did not produce a token")
                next_token = int(result.token_id)
                sequence.append(next_token)
            rows.append(sequence)
            state_snapshots.append(
                _slot_state_snapshot(
                    session,
                    slot=0,
                    live_count=len(prompt) + int(decode_tokens),
                )
            )
            session.reset()
    return rows, state_snapshots


def run(args: argparse.Namespace, argv: Sequence[str] | None = None) -> dict[str, Any]:
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if args.prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if args.steps_per_width <= 0:
        raise ValueError("steps_per_width must be positive")
    if args.max_layers <= 0:
        raise ValueError("max_layers must be positive")
    prompt_lengths = _parse_prompt_lengths(
        args.prompt_lengths,
        batch_size=args.batch_size,
        fallback_length=args.prompt_length,
    )
    if args.survivor_slot < 0 or args.survivor_slot >= args.batch_size:
        raise ValueError("survivor_slot must be within batch_size")
    if args.eos_slot is not None:
        if args.eos_slot < 0 or args.eos_slot >= args.batch_size:
            raise ValueError("eos_slot must be within batch_size")
        if args.eos_slot == args.survivor_slot:
            raise ValueError("eos_slot cannot be the surviving c=1 slot")
    total_decode_steps = int(args.batch_size) * int(args.steps_per_width)
    if args.max_sequence_length < max(prompt_lengths) + total_decode_steps + 1:
        raise ValueError("max_sequence_length must cover every shrinking decode step")

    os.environ.setdefault("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    compiler_version = _compiler_version(args.compiler_version_file)
    prompts = _load_ragged_prompt_slices(
        Path(args.fixture),
        prompt_lengths,
    )
    runner = Qwen35ParoNextTokenRunner(args.model)
    termination_order = _cancellation_order(
        args.batch_size,
        survivor_slot=args.survivor_slot,
    )
    expected_decode_counts = _decode_counts_for_order(
        args.batch_size,
        termination_order,
        steps_per_width=args.steps_per_width,
    )
    c1_sequences, c1_state_snapshots = _run_c1_reference(
        runner,
        prompts,
        expected_decode_counts,
        max_layers=args.max_layers,
        max_sequence_length=args.max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    eos_token_id = (
        None
        if args.eos_slot is None
        else int(c1_sequences[int(args.eos_slot)][expected_decode_counts[int(args.eos_slot)]])
    )
    eos_seed_collision = bool(
        args.eos_slot is not None
        and eos_token_id == int(c1_sequences[int(args.eos_slot)][0])
    )
    generated_by_request: dict[int, list[int]] = {}
    decoded_count_by_request: dict[int, int] = {}
    transition_trace: list[dict[str, Any]] = []
    batch_state_by_slot: dict[int, dict[str, Any]] = {}
    prefill_slab_trace: list[dict[str, Any]] = []
    eos_transition_observed = args.eos_slot is None
    lifecycle_widths_exact = True

    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=args.max_sequence_length,
        max_layers=args.max_layers,
        max_batch_size=args.batch_size,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        scheduler = ResidentBatchScheduler(capacity=args.batch_size)
        request_ids = [
            scheduler.submit(
                prompt,
                max_new_tokens=total_decode_steps,
                sampling=PerRowSamplingParams(
                    eos_token_id=eos_token_id if slot == args.eos_slot else None,
                ),
            )
            for slot, prompt in enumerate(prompts)
        ]
        admitted = scheduler.admit_pending()
        if tuple(request_ids) != tuple(admitted):
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
        slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(prompt_lengths),
            block_size=session.block_size,
        )
        if not slabs:
            raise RuntimeError("scheduler did not emit packed prefill slabs")
        seed_by_request: dict[int, int] = {}
        for slab in slabs:
            seed_results = session.prefill_native_packed(slab, sample=True)
            prefill_execution = dict(session.last_prefill_execution or {})
            prefill_slab_trace.append(
                {
                    "request_ids": list(slab.request_ids),
                    "physical_slots": list(slab.physical_slot_ids),
                    "prompt_lengths": [len(row) for row in slab.token_rows],
                    "rows": int(slab.rows),
                    "block_count": int(slab.block_count),
                    "linear_attention_prefill_path": prefill_execution.get(
                        "linear_attention_prefill_path"
                    ),
                    "full_attention_prefill_path": prefill_execution.get(
                        "full_attention_prefill_path"
                    ),
                    "blockers": list(prefill_execution.get("blockers", ())),
                }
            )
            for request_id, result in zip(slab.request_ids, seed_results, strict=True):
                if result is None:
                    raise RuntimeError("packed prefill did not produce a seed token")
                seed_by_request[int(request_id)] = int(result.token_id)
        if set(seed_by_request) != set(request_ids):
            raise RuntimeError("packed prefill did not cover every admitted request")
        next_token_by_request: dict[int, int] = {}
        for request_id in request_ids:
            token_id = seed_by_request[request_id]
            generated_by_request[request_id] = [token_id]
            decoded_count_by_request[request_id] = 0
            next_token_by_request[request_id] = token_id

        for stage_index, live_width in enumerate(range(args.batch_size, 0, -1)):
            stage_slots: list[list[int]] = []
            stage_request_ids: list[list[int]] = []
            completed_request_ids: list[int] = []
            termination_slot = termination_order[stage_index] if live_width > 1 else None
            termination_request = None if termination_slot is None else request_ids[termination_slot]
            termination_kind = (
                None
                if termination_slot is None
                else "eos"
                if termination_slot == args.eos_slot
                else "cancel"
            )
            for step_index in range(args.steps_per_width):
                work = scheduler.next_decode_work()
                if work is None:
                    raise RuntimeError("scheduler did not emit shrinking decode work")
                active_ids = tuple(
                    request_id
                    for request_id in work.request_ids
                    if request_id in next_token_by_request
                )
                if len(active_ids) != live_width:
                    raise RuntimeError(
                        f"expected live width {live_width}, scheduler emitted {len(active_ids)}"
                    )
                slots = [scheduler.active_batch.slot_for(request_id) for request_id in active_ids]
                positions = [
                    scheduler.active_batch.requests[request_id].context_len
                    for request_id in active_ids
                ]
                stage_slots.append(slots)
                stage_request_ids.append(list(active_ids))
                step = (
                    session.step_batch_native
                    if args.decode_execution == "native"
                    else session.step_batch_serial
                )
                results = step(
                    [next_token_by_request[request_id] for request_id in active_ids],
                    positions=positions,
                    slots=slots,
                    sample=True,
                )
                generated: list[GeneratedToken] = []
                for request_id, result in zip(active_ids, results, strict=True):
                    if result is None:
                        raise RuntimeError("shrinking decode did not produce a token")
                    token_id = int(result.token_id)
                    generated_by_request[request_id].append(token_id)
                    decoded_count_by_request[request_id] += 1
                    next_token_by_request[request_id] = token_id
                    eos_finished = bool(
                        termination_kind == "eos"
                        and request_id == termination_request
                        and step_index == args.steps_per_width - 1
                        and token_id == eos_token_id
                    )
                    generated.append(GeneratedToken(request_id, token_id, finished=eos_finished))
                completed = scheduler.record_generated(generated)
                for done in completed:
                    completed_request_ids.append(int(done.request_id))
                    next_token_by_request.pop(int(done.request_id), None)

            retiring_slot = int(args.survivor_slot) if termination_slot is None else int(termination_slot)
            retiring_request = request_ids[retiring_slot]
            batch_state_by_slot[retiring_slot] = _slot_state_snapshot(
                session,
                slot=retiring_slot,
                live_count=len(prompts[retiring_slot]) + decoded_count_by_request[retiring_request],
            )

            finish_reason: str | None = None
            finish_details: dict[str, Any] | None = None
            if termination_kind == "cancel":
                assert termination_request is not None
                done = scheduler.cancel(termination_request)
                if done is None:
                    raise RuntimeError(f"failed to cancel request {termination_request}")
                next_token_by_request.pop(termination_request, None)
                completed_request_ids.append(termination_request)
                finish_reason = done.finish_reason
                finish_details = done.finish_details.to_json_dict()
            elif termination_kind == "eos":
                assert termination_request is not None
                done = scheduler.completed.get(termination_request)
                eos_transition_observed = bool(
                    done is not None
                    and done.finish_reason == "stop"
                    and done.generated_tokens
                    and int(done.generated_tokens[-1]) == eos_token_id
                )
                if done is None:
                    fallback = scheduler.cancel(termination_request)
                    if fallback is not None:
                        next_token_by_request.pop(termination_request, None)
                        completed_request_ids.append(termination_request)
                        done = fallback
                if done is not None:
                    finish_reason = done.finish_reason
                    finish_details = done.finish_details.to_json_dict()

            expected_post_width = live_width - 1 if live_width > 1 else 0
            actual_post_width = scheduler.active_count
            if actual_post_width != expected_post_width:
                lifecycle_widths_exact = False
            transition_trace.append(
                {
                    "live_width": live_width,
                    "request_ids": stage_request_ids,
                    "physical_slots": stage_slots,
                    "termination_kind": termination_kind,
                    "terminated_request_id": termination_request,
                    "terminated_slot": termination_slot,
                    "completed_request_ids": completed_request_ids,
                    "finish_reason": finish_reason,
                    "finish_details": finish_details,
                    "expected_post_width": expected_post_width,
                    "actual_post_width": actual_post_width,
                }
            )

        if set(batch_state_by_slot) != set(range(args.batch_size)):
            raise RuntimeError("lifecycle gate did not snapshot every physical slot at retirement")

    decode_counts = [decoded_count_by_request[request_id] for request_id in request_ids]
    batch_state_snapshots = [batch_state_by_slot[slot] for slot in range(args.batch_size)]
    batch_sequences = [generated_by_request[request_id] for request_id in request_ids]
    row_equality = [batch == c1 for batch, c1 in zip(batch_sequences, c1_sequences, strict=True)]
    state_mismatches = [
        _state_snapshot_mismatches(batch, c1)
        for batch, c1 in zip(batch_state_snapshots, c1_state_snapshots, strict=True)
    ]
    state_row_equality = [not mismatches for mismatches in state_mismatches]
    decode_counts_exact = decode_counts == expected_decode_counts
    eos_transition_passed = bool(eos_transition_observed and not eos_seed_collision)
    passed = bool(
        all(row_equality)
        and all(state_row_equality)
        and decode_counts_exact
        and lifecycle_widths_exact
        and eos_transition_passed
    )
    payload: dict[str, Any] = {
        "schema": 1,
        "status": "eq_ok" if passed else "mismatch_found",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": f"qwen35_paro_shrinking_{args.decode_execution}_decode_gate",
        "command": _command(argv),
        "performance_claim": False,
        "hardware": _hardware_context(),
        "software": _software_context(),
        "workload": {
            "model": str(args.model),
            "fixture": str(args.fixture),
            "prompt_length": prompt_lengths[0] if len(set(prompt_lengths)) == 1 else None,
            "prompt_lengths": list(prompt_lengths),
            "ragged_contexts": len(set(prompt_lengths)) > 1,
            "initial_batch_size": int(args.batch_size),
            "steps_per_width": int(args.steps_per_width),
            "decode_tokens_by_request": decode_counts,
            "expected_decode_tokens_by_request": expected_decode_counts,
            "max_layers": int(args.max_layers),
            "max_sequence_length": int(args.max_sequence_length),
            "decode_execution": str(args.decode_execution),
            "native_compact_prefill": True,
            "prefill_slabs": prefill_slab_trace,
            "survivor_slot": int(args.survivor_slot),
            "termination_order": list(termination_order),
            "eos_slot": args.eos_slot,
            "eos_token_id": eos_token_id,
            "transitions": transition_trace,
        },
        "correctness": {
            "oracle": "independent c1 token sequence plus SHA-256 of all persistent linear state and full-attention KV prefixes",
            "passed": passed,
            "generated_token_equality": {
                "passed": all(row_equality),
                "row_equality": row_equality,
            },
            "persistent_state_identity": {
                "passed": all(state_row_equality),
                "row_equality": state_row_equality,
                "mismatch_components_by_slot": state_mismatches,
                "batch_aggregate_sha256_by_slot": [
                    snapshot["aggregate_sha256"] for snapshot in batch_state_snapshots
                ],
                "c1_aggregate_sha256_by_slot": [
                    snapshot["aggregate_sha256"] for snapshot in c1_state_snapshots
                ],
                "linear_layer_count": len(c1_state_snapshots[0]["linear"]),
                "full_attention_layer_count": len(c1_state_snapshots[0]["full_kv"]),
            },
            "decode_counts_exact": decode_counts_exact,
            "lifecycle_widths_exact": lifecycle_widths_exact,
            "no_group_wide_cancellation": lifecycle_widths_exact,
            "eos_transition": {
                "requested": args.eos_slot is not None,
                "observed": eos_transition_observed,
                "seed_collision": eos_seed_collision,
                "passed": eos_transition_passed,
            },
            "batch_sequences": batch_sequences,
            "c1_sequences": c1_sequences,
        },
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_payload_json(payload) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument(
        "--prompt-lengths",
        default=None,
        help="Optional comma-separated per-row prompt lengths; overrides --prompt-length.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-width", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument(
        "--survivor-slot",
        type=int,
        default=0,
        help="Physical slot that remains resident through the final c=1 stage.",
    )
    parser.add_argument(
        "--eos-slot",
        type=int,
        default=None,
        help="Optional retiring slot whose final c1-oracle token is configured and recorded as EOS.",
    )
    parser.add_argument(
        "--decode-execution",
        choices=("serial", "native"),
        default="serial",
        help="Use the exact serial bridge or the explicit native-width diagnostic.",
    )
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args, argv)
    print(_payload_json(payload))
    return 0 if payload["status"] == "eq_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
