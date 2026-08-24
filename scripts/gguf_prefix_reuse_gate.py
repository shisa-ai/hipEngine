#!/usr/bin/env python3
"""Gate GGUF prefix reuse against an independent full prefill.

The candidate goes through ``Qwen35GGUFResidentModelRunner`` with
``prefix_cache=radix``.  The source may remain active or be normally released
before continuation admission.  The continuation shares exact page-aligned
KV, restores live or cache-owned Conv/GDN state, and executes only its non-empty
suffix.  A private session then rebuilds the same continuation as the reference.
The gate compares deterministic output, every Conv/GDN state byte, logical
block-table-ordered live K/V bytes, lifecycle/refcounts, and matched-context
teacher-forced decode logits/state. Greedy and forced-token processed-argmax
routes are supported; stochastic sampling is intentionally outside this gate.

This is a correctness/lifecycle diagnostic, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
_HARDWARE_LABELS = {
    "hip_gfx1100": "AMD Radeon Pro W7900 (gfx1100)",
    "hip_gfx1151": "AMD Radeon 8060S (gfx1151)",
}


@contextmanager
def _temporary_env(updates: Mapping[str, str]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _logical_page_segments(
    allocation: Any,
    *,
    position: int,
    row_nbytes: int,
    block_size: int = 256,
) -> tuple[tuple[int, int], ...]:
    """Return backing-relative byte ranges in logical block-table order."""

    live_positions = int(position)
    row_bytes = int(row_nbytes)
    page_tokens = int(block_size)
    if live_positions < 0:
        raise ValueError("position must be non-negative")
    if row_bytes <= 0:
        raise ValueError("row_nbytes must be positive")
    if page_tokens <= 0:
        raise ValueError("block_size must be positive")
    required_pages = (live_positions + page_tokens - 1) // page_tokens
    block_ids = tuple(int(block_id) for block_id in allocation.block_ids)
    if len(block_ids) < required_pages:
        raise ValueError(
            f"device KV allocation does not cover {live_positions} live positions"
        )
    chunk_start = int(allocation.chunk_start_block_id)
    segments: list[tuple[int, int]] = []
    remaining = live_positions
    for block_id in block_ids[:required_pages]:
        local_page = block_id - chunk_start
        if local_page < 0:
            raise ValueError("device KV block precedes its backing chunk")
        rows = min(page_tokens, remaining)
        segments.append(
            (
                local_page * page_tokens * row_bytes,
                rows * row_bytes,
            )
        )
        remaining -= rows
    return tuple(segments)


def _fingerprint(raw: bytes) -> dict[str, Any]:
    return {
        "nbytes": len(raw),
        "blake2b_128": hashlib.blake2b(raw, digest_size=16).hexdigest(),
    }


def _copy_device_bytes(session: Any, buffer: Any) -> bytes:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(buffer.nbytes)
    raw = np.empty((size,), dtype=np.uint8)
    if size:
        copy_device_to_host(
            host_array_ptr(raw),
            DeviceBuffer(int(buffer.ptr), size),
            size,
            runtime=session.runtime,
        )
    return raw.tobytes()


def _copy_logical_kv_bytes(
    session: Any,
    buffer: Any,
    allocation: Any,
    *,
    position: int,
    row_nbytes: int,
) -> bytes:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    parts: list[bytes] = []
    for byte_offset, size in _logical_page_segments(
        allocation,
        position=position,
        row_nbytes=row_nbytes,
    ):
        if byte_offset + size > int(buffer.nbytes):
            raise ValueError("logical KV page range exceeds its backing cache")
        raw = np.empty((size,), dtype=np.uint8)
        if size:
            copy_device_to_host(
                host_array_ptr(raw),
                DeviceBuffer(int(buffer.ptr) + int(byte_offset), int(size)),
                int(size),
                runtime=session.runtime,
            )
        parts.append(raw.tobytes())
    return b"".join(parts)


def _capture_state(session: Any) -> dict[str, Any]:
    from hipengine.core.dtype import DType

    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    allocation = session.device_kv_allocation
    if allocation is None:
        raise RuntimeError("GGUF prefix gate requires a bound device KV allocation")
    session.runtime.device_synchronize()
    scratch = session.scratch
    cfg = session.runner.weights.config
    linear: list[dict[str, Any]] = []
    for layer_id, (conv, recurrent) in enumerate(
        zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
    ):
        if conv is None or recurrent is None:
            continue
        linear.append(
            {
                "layer": int(layer_id),
                "conv": _fingerprint(_copy_device_bytes(session, conv)),
                "recurrent": _fingerprint(_copy_device_bytes(session, recurrent)),
            }
        )

    live_positions = int(session.position)
    kv_row_nbytes = (
        int(cfg.head_count_kv)
        * int(cfg.key_length)
        * DType.BF16.itemsize
    )
    kv: list[dict[str, Any]] = []
    for layer_id, (key, value) in enumerate(
        zip(scratch.full_key_caches, scratch.full_value_caches, strict=True)
    ):
        if key is None or value is None:
            continue
        key_raw = _copy_logical_kv_bytes(
            session,
            key,
            allocation,
            position=live_positions,
            row_nbytes=kv_row_nbytes,
        )
        value_raw = _copy_logical_kv_bytes(
            session,
            value,
            allocation,
            position=live_positions,
            row_nbytes=kv_row_nbytes,
        )
        kv.append(
            {
                "layer": int(layer_id),
                "key": _fingerprint(key_raw),
                "value": _fingerprint(value_raw),
                "checked_nbytes": len(key_raw),
            }
        )
    return {
        "position": live_positions,
        "linear": linear,
        "kv": kv,
    }


def _layer_map(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["layer"]): row for row in rows}


def _compare_states(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if int(candidate["position"]) != int(reference["position"]):
        mismatches.append(
            {
                "component": "position",
                "layer": None,
                "part": None,
                "candidate": int(candidate["position"]),
                "reference": int(reference["position"]),
            }
        )
    for component, parts in (
        ("linear", ("conv", "recurrent")),
        ("kv", ("key", "value", "checked_nbytes")),
    ):
        candidate_layers = _layer_map(candidate[component])
        reference_layers = _layer_map(reference[component])
        for layer in sorted(set(candidate_layers) | set(reference_layers)):
            candidate_row = candidate_layers.get(layer, {})
            reference_row = reference_layers.get(layer, {})
            for part in parts:
                candidate_value = candidate_row.get(part)
                reference_value = reference_row.get(part)
                if candidate_value != reference_value:
                    mismatches.append(
                        {
                            "component": component,
                            "layer": int(layer),
                            "part": part,
                            "candidate": candidate_value,
                            "reference": reference_value,
                        }
                    )
    return mismatches


def _summarize_mismatches(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('component')}.{row.get('part')}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "count": len(rows),
        "component_counts": dict(sorted(counts.items())),
        "first": None if not rows else dict(rows[0]),
    }


def _lifecycle_exact(
    source_lifecycle: str,
    *,
    source_refcount_before_release: int,
    source_refcount_after_release: int,
    shared_refcount_after_admission: int,
    shared_refcount_after_continuation_release: int,
    final_refcounted_pages: int,
    final_pinned_pages: int = 0,
    workspace_lease_pages: int = 0,
    source_session_reset: bool,
    snapshot_evicted: bool,
) -> bool:
    if source_lifecycle == "active":
        return bool(
            source_refcount_before_release == 2
            and source_refcount_after_release == 1
            and shared_refcount_after_admission == 2
            and shared_refcount_after_continuation_release == 0
            and final_refcounted_pages == workspace_lease_pages
            and final_pinned_pages == workspace_lease_pages
        )
    if source_lifecycle == "completed":
        return bool(
            source_refcount_before_release == 1
            and source_refcount_after_release == 1
            and shared_refcount_after_admission == 2
            and shared_refcount_after_continuation_release == 1
            and final_refcounted_pages == workspace_lease_pages
            and final_pinned_pages == workspace_lease_pages
            and source_session_reset
            and snapshot_evicted
        )
    raise ValueError(f"unsupported source_lifecycle {source_lifecycle!r}")


def _production_metadata_exact(
    source_lifecycle: str,
    *,
    boundary: int,
    reused_tokens: int,
    source_request_id: int | None,
    source_id: int,
    clone_bytes: int,
    snapshot_hit: bool,
) -> bool:
    if source_lifecycle == "active":
        source_exact = source_request_id == int(source_id) and not snapshot_hit
    elif source_lifecycle == "completed":
        source_exact = source_request_id is None and snapshot_hit
    else:
        raise ValueError(f"unsupported source_lifecycle {source_lifecycle!r}")
    return bool(
        int(reused_tokens) == int(boundary)
        and int(clone_bytes) > 0
        and source_exact
    )


def _request(
    prompt: tuple[int, ...],
    *,
    max_tokens: int,
    sampler_mode: str,
    forced_token_id: int,
) -> Any:
    from hipengine.generation.registry import GenerationRequest

    processed_argmax = sampler_mode == "processed_argmax"
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        forced_tokens_pending=(
            (int(forced_token_id), int(forced_token_id) + 1)
            if processed_argmax
            else ()
        ),
        forced_token_reason=("tool_choice_required" if processed_argmax else None),
    )


def _prefill_work(request_id: int, tokens: tuple[int, ...]) -> Any:
    from hipengine.dispatch import WorkItem, WorkKind

    return WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=(int(request_id),),
        row_to_request=(int(request_id),),
        token_rows=(tokens,),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine import LLM
    from hipengine.benchmark.correctness import evaluate_logits

    boundary = int(args.prefix_tokens)
    suffix_tokens = int(args.suffix_tokens)
    teacher_steps = int(args.teacher_forced_steps)
    source_lifecycle = str(args.source_lifecycle)
    sampler_mode = str(args.sampler_mode)
    if boundary <= 0 or boundary % 256:
        raise ValueError("prefix_tokens must be a positive multiple of 256")
    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    if teacher_steps <= 0:
        raise ValueError("teacher_forced_steps must be positive")
    full_length = boundary + suffix_tokens
    required_positions = full_length + teacher_steps
    max_sequence_length = int(args.max_sequence_length)
    if required_positions > max_sequence_length:
        raise ValueError("max_sequence_length does not cover the correctness trajectory")

    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    prefix = (int(args.prefix_token_id),) * boundary
    suffix = (int(args.suffix_token_id),) * suffix_tokens
    continued_prompt = (*prefix, *suffix)
    source_id = 1001
    continuation_id = 1002
    oracle_id = 1003

    llm = LLM(
        str(model),
        backend=str(args.backend),
        quant=str(args.quant),
        max_active_requests=3,
        prefix_cache="radix",
    )
    runner = None
    pool = None
    source_state = SimpleNamespace(request_id=source_id)
    continuation_state = SimpleNamespace(request_id=continuation_id)
    oracle_lease = None
    oracle_pool = None
    oracle_allocation = None
    oracle_bound = False
    source_released = False
    continuation_released = False
    source_refcount_before_release = -1
    source_refcount_after_release = -1
    shared_refcount_after_admission = -1
    shared_refcount_after_continuation_release = -1
    source_session_reset = False
    snapshot_evicted = False
    payload: dict[str, Any] | None = None
    try:
        llm.prepare(max_sequence_length=max_sequence_length)
        wrapper = llm._get_text_generator()
        runner = wrapper._runner
        pool = runner.kv_pool
        if pool is None:
            raise RuntimeError("GGUF resident runner did not configure a device KV pool")

        source_request = _request(
            prefix,
            max_tokens=teacher_steps + 2,
            sampler_mode=sampler_mode,
            forced_token_id=int(args.forced_token_id),
        )
        runner.register_batch((source_id,), source_request, prompt_rows=(prefix,))
        runner.reserve_admission(source_state)
        runner.prefill_batch(_prefill_work(source_id, prefix), commit=True)
        source_row = runner._rows[source_id]
        if source_row.lease is None or source_row.slot is None or source_row.kv_allocation is None:
            raise RuntimeError("GGUF prefix source did not retain complete resident state")
        source_session = source_row.lease.session
        source_allocation = source_row.kv_allocation
        source_before = _capture_state(source_session)
        source_token = int(source_row.slot.prev_token)
        source_block_ids = tuple(
            int(block_id) for block_id in source_allocation.block_ids
        )
        if not source_block_ids:
            raise RuntimeError("GGUF prefix source has no device KV blocks")
        if source_lifecycle == "completed":
            source_refcount_before_release = int(pool.refcount(source_block_ids[0]))
            runner._release_row_resources(
                source_row,
                retain_prefix_snapshots=True,
            )
            runner._rows.pop(source_id)
            source_released = True
            source_refcount_after_release = int(pool.refcount(source_block_ids[0]))
            source_session_reset = bool(
                int(source_session.position) == 0
                and source_session.device_kv_allocation is None
                and any(lease.session is source_session for lease in runner._available)
            )

        continuation_request = _request(
            continued_prompt,
            max_tokens=teacher_steps + 1,
            sampler_mode=sampler_mode,
            forced_token_id=int(args.forced_token_id),
        )
        runner.register_batch(
            (continuation_id,),
            continuation_request,
            prompt_rows=(continued_prompt,),
        )
        runner.reserve_admission(continuation_state)
        continuation_row = runner._rows[continuation_id]
        if continuation_row.lease is None or continuation_row.kv_allocation is None:
            raise RuntimeError("GGUF prefix continuation was not admitted")
        continuation_session = continuation_row.lease.session
        continuation_allocation = continuation_row.kv_allocation
        shared_blocks = tuple(
            int(block_id)
            for block_id in continuation_allocation.reused_block_ids
        )
        if not shared_blocks:
            raise RuntimeError("GGUF prefix continuation did not reuse a device KV page")
        shared_refcount_after_admission = int(pool.refcount(shared_blocks[0]))
        candidate_boundary = _capture_state(continuation_session)
        clone_boundary_mismatches = _compare_states(candidate_boundary, source_before)
        runner.prefill_batch(_prefill_work(continuation_id, prefix), commit=True)
        runner.prefill_batch(_prefill_work(continuation_id, suffix), commit=True)
        if continuation_row.slot is None:
            raise RuntimeError("GGUF prefix continuation did not finish suffix prefill")
        continuation_token = int(continuation_row.slot.prev_token)
        candidate_initial = _capture_state(continuation_session)
        if source_lifecycle == "active":
            source_after_candidate = _capture_state(source_session)
            source_immutability_mismatches = _compare_states(
                source_after_candidate,
                source_before,
            )
            source_immutability_applicable = True
        else:
            source_immutability_mismatches = []
            source_immutability_applicable = False

        pages = (required_positions + 255) // 256
        if not runner._available:
            raise RuntimeError("GGUF prefix gate has no private oracle session")
        oracle_lease = runner._available[-1]
        oracle_pool = oracle_lease.session.create_device_kv_pool(
            initial_pages=pages,
            low_water_pages=pages,
            high_water_pages=pages,
            chunk_pages=pages,
            idle_grace_seconds=30.0,
        )
        oracle_allocation = oracle_pool.allocate(
            oracle_id,
            pages,
            now_seconds=time.monotonic(),
        )
        oracle_lease.session.bind_device_kv_allocation(
            oracle_pool,
            oracle_allocation,
        )
        oracle_bound = True
        if not runner._available or runner._available[-1] is not oracle_lease:
            raise RuntimeError("GGUF oracle session order changed during private admission")
        runner._available.pop()
        oracle_session = oracle_lease.session

        # Keep the one-shot bulk route as a diagnostic because its row-count
        # arithmetic can differ from scheduler chunking.  The gating reference
        # privately processes every token through the exact same 256+suffix
        # production chunks that prefix reuse replaces.
        one_shot_applicable = len(continued_prompt) <= 8192
        one_shot_token = None
        one_shot_state_mismatches: list[dict[str, Any]] = []
        if one_shot_applicable:
            one_shot_result = oracle_session.prefill(
                continued_prompt,
                return_logits=False,
            )
            one_shot_token = int(one_shot_result.token_id)
            one_shot_state = _capture_state(oracle_session)
            one_shot_state_mismatches = _compare_states(
                candidate_initial,
                one_shot_state,
            )
            oracle_session.reset()
        with _temporary_env({_CAPTURE_PREFILL_GDN_ENV: "1"}):
            oracle_session.prefill_batch_native(
                [prefix],
                sessions=[oracle_session],
                full_prompt_lengths=[len(continued_prompt)],
                return_logits=False,
                return_hidden_seeds=False,
            )
            scheduler_boundary = _capture_state(oracle_session)
            scheduler_boundary_mismatches = _compare_states(
                candidate_boundary,
                scheduler_boundary,
            )
            scheduler_results = oracle_session.prefill_batch_native(
                [suffix],
                sessions=[oracle_session],
                full_prompt_lengths=[len(continued_prompt)],
                return_logits=False,
                return_hidden_seeds=False,
            )
        if len(scheduler_results) != 1:
            raise RuntimeError("GGUF private chunked oracle returned the wrong result count")
        scheduler_token = int(scheduler_results[0].token_id)
        scheduler_state = _capture_state(oracle_session)
        scheduler_state_mismatches = _compare_states(candidate_initial, scheduler_state)

        # The retained GGUF correctness contract is independent c1 state/KV.
        # Rebuild the active source privately, then consume the unmatched
        # suffix one token at a time on the matched context.
        oracle_session.reset()
        semantic_prefix_result = oracle_session.prefill(prefix, return_logits=False)
        semantic_boundary = _capture_state(oracle_session)
        semantic_boundary_mismatches = _compare_states(
            candidate_boundary,
            semantic_boundary,
        )
        semantic_result = semantic_prefix_result
        for suffix_index, suffix_token in enumerate(suffix):
            semantic_result = oracle_session.step(
                int(suffix_token),
                return_logits=bool(
                    sampler_mode == "processed_argmax"
                    and suffix_index == len(suffix) - 1
                ),
            )
        if sampler_mode == "processed_argmax":
            from hipengine.generation.qwen35_gguf import (
                _gguf_row_sampling_state,
                _request_with_tokenizer_eos,
                _select_from_gguf_logits,
            )

            oracle_request = _request_with_tokenizer_eos(
                continuation_request,
                runner.generator.tokenizer,
            )
            oracle_sampling_state = _gguf_row_sampling_state(
                oracle_request,
                list(continued_prompt),
                row_index=0,
            )
            oracle_sample = _select_from_gguf_logits(
                semantic_result,
                oracle_request,
                oracle_sampling_state,
            )
            oracle_token = int(oracle_sample.token_id)
        else:
            oracle_token = int(semantic_result.token_id)
        reference_initial = _capture_state(oracle_session)
        initial_state_mismatches = _compare_states(candidate_initial, reference_initial)

        if source_lifecycle == "active":
            source_refcount_before_release = int(pool.refcount(shared_blocks[0]))
            runner.rollback_admission(source_state)
            source_released = True
            source_refcount_after_release = int(pool.refcount(shared_blocks[0]))
            source_session_reset = bool(
                int(source_session.position) == 0
                and source_session.device_kv_allocation is None
                and any(lease.session is source_session for lease in runner._available)
            )

        candidate_logits: list[np.ndarray] = []
        reference_logits: list[np.ndarray] = []
        candidate_predicted = [continuation_token]
        reference_predicted = [oracle_token]
        forced_tokens: list[int] = []
        forced_token = oracle_token
        for _ in range(teacher_steps):
            forced_tokens.append(int(forced_token))
            candidate_result = continuation_session.step(
                int(forced_token),
                return_logits=True,
            )
            reference_result = oracle_session.step(
                int(forced_token),
                return_logits=True,
            )
            candidate_logits.append(
                np.ascontiguousarray(candidate_result.logits, dtype=np.float32).copy()
            )
            reference_logits.append(
                np.ascontiguousarray(reference_result.logits, dtype=np.float32).copy()
            )
            if sampler_mode == "processed_argmax":
                if (
                    continuation_row.sampling_request is None
                    or continuation_row.sampling_state is None
                ):
                    raise RuntimeError(
                        "processed-argmax candidate lost its sampling state"
                    )
                candidate_sample = _select_from_gguf_logits(
                    candidate_result,
                    continuation_row.sampling_request,
                    continuation_row.sampling_state,
                )
                reference_sample = _select_from_gguf_logits(
                    reference_result,
                    oracle_request,
                    oracle_sampling_state,
                )
                candidate_next = int(candidate_sample.token_id)
                reference_next = int(reference_sample.token_id)
            else:
                candidate_next = int(candidate_result.token_id)
                reference_next = int(reference_result.token_id)
            candidate_predicted.append(candidate_next)
            reference_predicted.append(reference_next)
            forced_token = reference_next

        metrics = evaluate_logits(
            np.stack(reference_logits, axis=0),
            np.stack(candidate_logits, axis=0),
        )
        candidate_final = _capture_state(continuation_session)
        reference_final = _capture_state(oracle_session)
        final_state_mismatches = _compare_states(candidate_final, reference_final)

        observability = runner.observability_snapshot()
        pool_before_final_release = pool.stats.to_json_dict()
        runner.rollback_admission(continuation_state)
        continuation_released = True
        shared_refcount_after_continuation_release = int(
            pool.refcount(shared_blocks[0])
        )
        pool_after_continuation_release = pool.stats.to_json_dict()
        if source_lifecycle == "completed":
            snapshot_evicted = bool(runner._evict_prefix_snapshot(prefix))
        oracle_session.reset()
        detached = oracle_session.unbind_device_kv_allocation()
        if detached is not oracle_allocation:
            raise RuntimeError("GGUF oracle detached a different KV allocation")
        released = oracle_pool.release(oracle_id, now_seconds=time.monotonic())
        if released is not oracle_allocation:
            raise RuntimeError("GGUF pool released a different oracle allocation")
        oracle_bound = False
        oracle_pool.close()
        oracle_pool = None
        runner._available.append(oracle_lease)
        final_pool = pool.stats.to_json_dict()
        pool_memory = runner.kv_pool_memory_snapshot()
        workspace_lease_pages = int(
            pool_memory.get("packed_workspace_lease_pages", 0)
        )

        output_exact = continuation_token == oracle_token
        trajectory_exact = candidate_predicted == reference_predicted
        candidate_sampler_mode = (
            "unknown"
            if continuation_row.sampler_plan is None
            else str(continuation_row.sampler_plan.mode.value)
        )
        sampler_route_exact = bool(
            candidate_sampler_mode == sampler_mode
            and (
                sampler_mode != "processed_argmax"
                or (
                    continuation_token == int(args.forced_token_id)
                    and continuation_row.full_vocab_logits_d2h is True
                    and int(continuation_row.logits_d2h_bytes or 0) > 0
                )
            )
        )
        initial_state_exact = not initial_state_mismatches
        final_state_exact = not final_state_mismatches
        source_immutable = not source_immutability_mismatches
        lifecycle_exact = _lifecycle_exact(
            source_lifecycle,
            source_refcount_before_release=source_refcount_before_release,
            source_refcount_after_release=source_refcount_after_release,
            shared_refcount_after_admission=shared_refcount_after_admission,
            shared_refcount_after_continuation_release=(
                shared_refcount_after_continuation_release
            ),
            final_refcounted_pages=int(final_pool["refcounted_pages"]),
            final_pinned_pages=int(final_pool["pinned_pages"]),
            workspace_lease_pages=workspace_lease_pages,
            source_session_reset=source_session_reset,
            snapshot_evicted=snapshot_evicted,
        )
        production_metadata_exact = _production_metadata_exact(
            source_lifecycle,
            boundary=boundary,
            reused_tokens=int(continuation_row.prefix_reused_tokens),
            source_request_id=continuation_row.prefix_source_request_id,
            source_id=source_id,
            clone_bytes=int(continuation_row.prefix_state_clone_bytes),
            snapshot_hit=bool(continuation_row.prefix_snapshot_hit),
        )
        clone_boundary_exact = not clone_boundary_mismatches
        semantic_boundary_exact = not semantic_boundary_mismatches
        passed = bool(
            clone_boundary_exact
            and semantic_boundary_exact
            and output_exact
            and trajectory_exact
            and initial_state_exact
            and final_state_exact
            and source_immutable
            and lifecycle_exact
            and production_metadata_exact
            and sampler_route_exact
            and metrics.passed
        )
        payload = {
            "schema": 1,
            "kind": (
                "gguf_active_prefix_reuse_correctness_gate"
                if source_lifecycle == "active"
                else "gguf_completed_prefix_reuse_correctness_gate"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "passed": passed,
            "correctness_claim": True,
            "performance_claim": False,
            "model": str(model),
            "quant": str(args.quant),
            "backend": str(args.backend),
            "target_arch": str(runner.generator.target_arch),
            "hardware": str(
                args.hardware_label
                or _HARDWARE_LABELS.get(str(args.backend), str(args.backend))
            ),
            "workload": {
                "prefix_token_id": int(args.prefix_token_id),
                "prefix_tokens": boundary,
                "suffix_token_id": int(args.suffix_token_id),
                "suffix_tokens": suffix_tokens,
                "teacher_forced_steps": teacher_steps,
                "max_sequence_length": max_sequence_length,
                "sampling": (
                    "forced_sequence_processed_argmax_then_reference_teacher_forced"
                    if sampler_mode == "processed_argmax"
                    else "greedy_top1_then_reference_teacher_forced"
                ),
                "kv_dtype": "bf16",
                "source_lifecycle": source_lifecycle,
                "processor_forced_token_ids": (
                    [int(args.forced_token_id), int(args.forced_token_id) + 1]
                    if sampler_mode == "processed_argmax"
                    else []
                ),
            },
            "production_route": {
                "prefix_cache_mode": "radix",
                "requested_sampler_mode": sampler_mode,
                "candidate_sampler_mode": candidate_sampler_mode,
                "sampler_route_exact": sampler_route_exact,
                "full_vocab_logits_d2h": continuation_row.full_vocab_logits_d2h,
                "logits_d2h_bytes": continuation_row.logits_d2h_bytes,
                "prefix_reused_tokens": int(continuation_row.prefix_reused_tokens),
                "prefix_source_request_id": continuation_row.prefix_source_request_id,
                "prefix_state_clone_bytes": int(continuation_row.prefix_state_clone_bytes),
                "prefix_snapshot_hit": bool(continuation_row.prefix_snapshot_hit),
                "source_block_ids": list(source_block_ids),
                "continuation_block_ids": [
                    int(block_id) for block_id in continuation_allocation.block_ids
                ],
                "continuation_reused_block_ids": list(shared_blocks),
                "oracle_block_ids": [int(block_id) for block_id in oracle_allocation.block_ids],
                "metadata_exact": production_metadata_exact,
            },
            "prefill_oracle": {
                "route": "private_active_source_then_c1_teacher_forced_suffix",
                "source_predicted_token_id": source_token,
                "semantic_source_predicted_token_id": int(semantic_prefix_result.token_id),
                "candidate_predicted_token_id": continuation_token,
                "reference_predicted_token_id": oracle_token,
                "output_exact": output_exact,
                "clone_boundary_exact": clone_boundary_exact,
                "clone_boundary_mismatches": clone_boundary_mismatches,
                "semantic_boundary_exact": semantic_boundary_exact,
                "semantic_boundary_mismatches": semantic_boundary_mismatches,
                "initial_state_exact": initial_state_exact,
                "initial_state_mismatches": initial_state_mismatches,
                "source_immutable": source_immutable,
                "source_immutability_applicable": source_immutability_applicable,
                "source_immutability_mismatches": source_immutability_mismatches,
                "one_shot_bulk_diagnostic": {
                    "applicable": one_shot_applicable,
                    "skipped_reason": (
                        None
                        if one_shot_applicable
                        else "diagnostic_bulk_prefill_limit_8192"
                    ),
                    "predicted_token_id": one_shot_token,
                    "output_exact": (
                        continuation_token == one_shot_token
                        if one_shot_applicable
                        else None
                    ),
                    "state_exact": (
                        not one_shot_state_mismatches
                        if one_shot_applicable
                        else None
                    ),
                    "state_mismatch_summary": _summarize_mismatches(
                        one_shot_state_mismatches
                    ),
                    "gating": False,
                },
                "scheduler_chunk_diagnostic": {
                    "predicted_token_id": scheduler_token,
                    "output_exact": continuation_token == scheduler_token,
                    "boundary_state_mismatch_summary": _summarize_mismatches(
                        scheduler_boundary_mismatches
                    ),
                    "final_state_mismatch_summary": _summarize_mismatches(
                        scheduler_state_mismatches
                    ),
                    "gating": False,
                },
            },
            "teacher_forced": {
                "forced_token_ids": forced_tokens,
                "candidate_predicted_token_ids": candidate_predicted,
                "reference_predicted_token_ids": reference_predicted,
                "candidate_response_token_ids": candidate_predicted,
                "reference_response_token_ids": reference_predicted,
                "trajectory_exact": trajectory_exact,
                "kl_mean": metrics.kl_mean,
                "kl_max": metrics.kl_max,
                "top1_agreement": metrics.top1_agreement,
                "gate_passed": metrics.passed,
                "final_state_exact": final_state_exact,
                "final_state_mismatches": final_state_mismatches,
            },
            "lifecycle": {
                "source_lifecycle": source_lifecycle,
                "source_released_before_continuation_admission": (
                    source_lifecycle == "completed"
                ),
                "source_session_reset": source_session_reset,
                "source_refcount_before_release": source_refcount_before_release,
                "source_refcount_after_release": source_refcount_after_release,
                "shared_refcount_after_admission": shared_refcount_after_admission,
                "shared_refcount_after_continuation_release": (
                    shared_refcount_after_continuation_release
                ),
                "snapshot_evicted": snapshot_evicted,
                "final_refcounted_pages": int(final_pool["refcounted_pages"]),
                "final_pinned_pages": int(final_pool["pinned_pages"]),
                "workspace_lease_pages": workspace_lease_pages,
                "final_request_refcounted_pages": int(
                    final_pool["refcounted_pages"]
                ) - workspace_lease_pages,
                "exact": lifecycle_exact,
            },
            "memory_accounting": {
                "page_bytes": int(pool.page_bytes),
                "reused_pages": len(shared_blocks),
                "saved_live_bytes": len(shared_blocks) * int(pool.page_bytes),
                "pool_before_final_release": pool_before_final_release,
                "pool_after_continuation_release": pool_after_continuation_release,
                "pool_after_snapshot_eviction": final_pool,
                "timing_claim": False,
            },
            "observability": observability,
            "notes": [
                "Every Conv/GDN byte and logical block-table-ordered live BF16 K/V byte is compared.",
                "The gating oracle independently rebuilds the active source and consumes the suffix through exact c1 steps.",
                "One-shot bulk and private scheduler-chunk output/state remain non-gating row-shape diagnostics.",
                (
                    "The source session is reset before continuation admission; cache-owned state is mandatory."
                    if source_lifecycle == "completed"
                    else "The live source is reclaimed before teacher-forced survivor decode."
                ),
                "Saved live bytes are allocator accounting, not a process-wide peak-memory claim.",
                "No timing from this correctness run is a performance claim.",
            ],
        }
    finally:
        if runner is not None and pool is not None:
            if not continuation_released:
                row = runner._rows.get(continuation_id)
                if row is not None and row.lease is not None:
                    runner.rollback_admission(continuation_state)
            if not source_released:
                row = runner._rows.get(source_id)
                if row is not None and row.lease is not None:
                    runner.rollback_admission(source_state)
            if (
                oracle_bound
                and oracle_lease is not None
                and oracle_pool is not None
                and oracle_allocation is not None
            ):
                oracle_lease.session.invalidate_device_kv_graphs()
                oracle_lease.session.reset()
                oracle_lease.session.unbind_device_kv_allocation()
                oracle_pool.release(oracle_id, now_seconds=time.monotonic())
                oracle_pool.close()
                runner._available.append(oracle_lease)
        llm.close()
    if payload is None:
        raise RuntimeError("GGUF prefix reuse gate produced no payload")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=("hip_gfx1100", "hip_gfx1151"),
        default="hip_gfx1151",
    )
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prefix-token-id", type=int, default=9707)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-token-id", type=int, default=9708)
    parser.add_argument("--suffix-tokens", type=int, default=1)
    parser.add_argument("--teacher-forced-steps", type=int, default=4)
    parser.add_argument(
        "--sampler-mode",
        choices=("greedy_fast", "processed_argmax"),
        default="greedy_fast",
    )
    parser.add_argument("--forced-token-id", type=int, default=9709)
    parser.add_argument(
        "--source-lifecycle",
        choices=("active", "completed"),
        default="active",
        help="Keep the source live or release it into cache ownership before admission",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--hardware-label")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    command_args = list(sys.argv[1:] if argv is None else argv)
    payload["command"] = shlex.join(
        [sys.executable, "scripts/gguf_prefix_reuse_gate.py", *command_args]
    )
    payload["environment"] = {
        key: os.environ.get(key)
        for key in (
            "HIPENGINE_COMPILER_VERSION_FILE",
            "HIPENGINE_GGUF_DECODE_REPACK",
            "HIPENGINE_GGUF_WMMA_PREFILL",
            "HIPENGINE_GGUF_GEMV_DECODE",
            "HIPENGINE_GGUF_GDN_PREFILL_MODE",
            "HIP_VISIBLE_DEVICES",
        )
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
