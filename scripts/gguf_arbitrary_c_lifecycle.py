#!/usr/bin/env python3
"""Validate arbitrary-C GGUF retirement and new admission against c1.

The gate keeps scheduler slot identity stable, lowers logical C through the
production resident runner, and compares generated tokens plus every Conv/GDN
state and live BF16 KV byte against independent c1 checkpoints.  It is a
correctness diagnostic, not a throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hipengine import LLM, SamplingParams  # noqa: E402
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.core.memory import memory_stats  # noqa: E402
from hipengine.generation import GenerationRequest  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
)
from scripts.gguf_packed_ar_state_oracle import (  # noqa: E402
    _capture_state,
    _compare_state_rows,
)


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
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


_DEFAULT_CERT_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)


def _resolve_widths() -> tuple[int, ...]:
    """Resolve the active shared-slot AR physical width set for the gate.

    Mirrors the owner's resolution: an explicit
    ``HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS`` override (comma/space
    separated) selects the widths under test; otherwise the production default
    (1..8, promoted 2026-08-20) is used. The mask/declared-width assertions
    below then follow the same set the owner routes with.
    """
    override = os.environ.get(
        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", ""
    ).strip()
    if not override:
        return _DEFAULT_CERT_WIDTHS
    widths = tuple(int(item) for item in override.replace(",", " ").split())
    if not widths or widths[0] != 1 or tuple(sorted(set(widths))) != widths:
        raise ValueError(
            "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS must be sorted unique widths starting at c1"
        )
    return widths


def _request(prompts: Sequence[Sequence[int]], *, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=tuple(tuple(int(token) for token in prompt) for prompt in prompts),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _run_reference(
    session: Any,
    prompt: Sequence[int],
    *,
    max_tokens: int,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    result = session.prefill(
        tuple(int(token) for token in prompt),
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    tokens = [int(result.token_id)]
    states = {1: _capture_state(session)}
    while len(tokens) < int(max_tokens):
        result = session.step(tokens[-1], return_logits=False)
        tokens.append(int(result.token_id))
        states[len(tokens)] = _capture_state(session)
    return tokens, states


def _compare_one(actual: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return _compare_state_rows([actual], [expected])


def _plan_summary(plan: Any) -> dict[str, Any] | None:
    if not isinstance(plan, dict) or not plan:
        return None
    return json.loads(json.dumps(plan))


def _poll_one(
    adapter: Any,
    runner: Any,
    *,
    label: str,
    token_counts: dict[int, int],
    timeline: list[dict[str, Any]],
) -> tuple[Any, ...]:
    events = tuple(adapter.poll(max_ticks=1))
    if not events:
        snapshot = adapter.live_loop_snapshot()
        debug_path = Path("/tmp/hipengine-gguf-arbitrary-c-stall.json")
        debug_path.write_text(
            json.dumps(
                {"label": str(label), "snapshot": snapshot, "timeline": timeline},
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"resident arbitrary-C loop stalled at {label}; debug={debug_path}"
        )
    for event in events:
        if event.kind == "token":
            token_counts[int(event.request_id)] = token_counts.get(int(event.request_id), 0) + 1
    work = [event.work_kind.value for event in events if event.kind == "work"]
    plan = (
        _plan_summary(runner._last_physical_group_plan)
        if work == ["decode"]
        else None
    )
    timeline.append(
        {
            "label": str(label),
            "work": work,
            "admitted": [int(event.request_id) for event in events if event.kind == "admitted"],
            "tokens": [int(event.request_id) for event in events if event.kind == "token"],
            "completed": [int(event.request_id) for event in events if event.kind == "completed"],
            "token_counts": {str(key): int(value) for key, value in sorted(token_counts.items())},
            "physical_group_plan": plan,
        }
    )
    return events


def _all_native(plan: dict[str, Any] | None) -> bool:
    def group_is_native(group: Mapping[str, Any]) -> bool:
        execution_path = group.get("execution_path")
        if execution_path == "packed_native":
            return True
        return bool(
            execution_path in {"native_c1_eager", "native_c1_graph"}
            and int(group.get("physical_rows", 0)) == 1
            and tuple(bool(value) for value in group.get("active_mask", ())) == (True,)
        )

    return bool(
        plan
        and int(plan.get("group_count", 0)) > 0
        and all(group_is_native(group) for group in plan.get("groups", ()))
    )


def _group_masks(plan: dict[str, Any] | None) -> list[str]:
    if plan is None:
        return []
    return [
        "".join("1" if active else "0" for active in group.get("active_mask", ()))
        for group in plan.get("groups", ())
    ]


def _expected_dense_group_masks(
    rows: int,
    buckets: Sequence[int] = _DEFAULT_CERT_WIDTHS,
    *,
    composition: Sequence[int] | None = None,
) -> list[str]:
    """Expected dense active masks for one round.

    When an explicit ``composition`` (e.g. the artifact-backed D2 partition) is
    supplied, each group is exactly that width and fully dense. Otherwise the
    ceiling (max-bucket) chunking with masked remainder is assumed.
    """
    buckets = tuple(int(bucket) for bucket in buckets)
    if composition is not None:
        return ["1" * int(width) for width in composition]
    remaining = int(rows)
    masks: list[str] = []
    while remaining > 0:
        active = min(buckets[-1], remaining)
        width = next(bucket for bucket in buckets if bucket >= active)
        masks.append("1" * active + "0" * (width - active))
        remaining -= active
    return masks


def _expected_hole_group_masks(
    rows: int,
    cancel_slots: Sequence[int],
    *,
    compact: bool,
    buckets: Sequence[int] = _DEFAULT_CERT_WIDTHS,
    composition: Sequence[int] | None = None,
) -> list[str]:
    buckets = tuple(int(bucket) for bucket in buckets)
    if compact:
        compact_rows = int(rows) - len(cancel_slots)
        compact_composition = _d2_composition(compact_rows)
        return _expected_dense_group_masks(
            compact_rows, buckets, composition=compact_composition
        )
    if composition is not None:
        masks = [list("1" * int(width)) for width in composition]
        offsets: list[int] = []
        cursor = 0
        for width in composition:
            offsets.append(cursor)
            cursor += int(width)
        for slot in cancel_slots:
            slot_i = int(slot)
            for width, base in zip(composition, offsets, strict=True):
                if base <= slot_i < base + int(width):
                    masks[offsets.index(base)][slot_i - base] = "0"
                    break
        return ["".join(mask) for mask in masks]
    masks = [list(mask) for mask in _expected_dense_group_masks(rows, buckets)]
    max_bucket = buckets[-1]
    for slot in cancel_slots:
        group_index, local_index = divmod(int(slot), max_bucket)
        masks[group_index][local_index] = "0"
    return ["".join(mask) for mask in masks]


def _d2_composition(rows: int) -> tuple[int, ...] | None:
    """Return the artifact-backed D2 composition for ``rows`` when D2 is active
    (``HIPENGINE_GGUF_AR_D2_COST_ARTIFACT`` set), else ``None`` (ceiling)."""
    path = os.environ.get("HIPENGINE_GGUF_AR_D2_COST_ARTIFACT", "").strip()
    if not path:
        return None
    from hipengine.dispatch.d2_resolver import cost_table_from_artifact, d2_partition

    cost_table = cost_table_from_artifact(path)
    return tuple(int(width) for width in d2_partition(int(rows), cost_table))


def _state_kv_accepted(*, bit_exact: bool, allow_c1_arithmetic_drift: bool) -> bool:
    return bool(bit_exact or allow_c1_arithmetic_drift)


def _tracked_memory_recovery(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_current = int(before.get("current_allocated_bytes", 0))
    after_current = int(after.get("current_allocated_bytes", 0))
    before_active = int(before.get("active_allocations", 0))
    after_active = int(after.get("active_allocations", 0))
    return {
        "scope": "hipengine_tracked_process",
        "passed": bool(
            after_current == before_current
            and after_active == before_active
        ),
        "current_allocated_bytes_before": before_current,
        "current_allocated_bytes_after": after_current,
        "current_allocated_delta_bytes": after_current - before_current,
        "active_allocations_before": before_active,
        "active_allocations_after": after_active,
        "active_allocation_delta": after_active - before_active,
        "peak_allocated_bytes": int(after.get("peak_allocated_bytes", 0)),
        "total_allocated_bytes": int(after.get("total_allocated_bytes", 0)),
        "total_freed_bytes": int(after.get("total_freed_bytes", 0)),
    }


def _load_quality_gate(
    path: Path,
    *,
    model: Path,
    backend: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    summary = payload.get("quality", {}).get("summary", {})
    protocol = payload.get("protocol", {})
    provenance = payload.get("provenance", {})
    if not (
        payload.get("kind")
        == "hipengine_execution_profile_gguf_batch_route_requalification_capture"
        and payload.get("measurement_valid") is True
        and payload.get("quality", {}).get("hard_gates_passed") is True
        and float(summary.get("kl_max", float("inf"))) <= 0.05
        and float(summary.get("top1_agreement", 0.0)) >= 0.90
        and protocol.get("route_profile") == "current_package_direct"
        and {3, 5, 6, 7}.issubset(
            {int(width) for width in protocol.get("static_widths", ())}
        )
        and provenance.get("dirty") is False
        and str(provenance.get("resolved_backend")) == str(backend)
        and Path(str(provenance.get("model_path", ""))).resolve() == model
    ):
        raise ValueError(f"quality artifact is not a valid matching hard gate: {resolved}")
    return {
        "path": str(resolved),
        "hipengine_commit": provenance.get("hipengine_commit"),
        "host_name": provenance.get("host_name"),
        "hard_gate_passed": True,
        "kl_max": float(summary["kl_max"]),
        "top1_agreement": float(summary["top1_agreement"]),
        "rows": int(summary["rows"]),
        "route_profile": str(protocol["route_profile"]),
        "static_widths": [int(width) for width in protocol["static_widths"]],
    }


def _row_resource_identity(row: Any) -> dict[str, Any]:
    lease = row.lease
    if lease is None:
        raise RuntimeError("active row has no resident session lease")
    session = lease.session
    scratch = session.scratch
    if scratch is None:
        raise RuntimeError("active row resident session is closed")
    allocation = row.kv_allocation
    return {
        "session_id": id(session),
        "allocation_id": None if allocation is None else id(allocation),
        "block_ids": (
            [] if allocation is None else [int(block) for block in allocation.block_ids]
        ),
        "conv_ptrs": [
            None if buffer is None else int(buffer.ptr)
            for buffer in scratch.layer_conv_states
        ],
        "recurrent_ptrs": [
            None if buffer is None else int(buffer.ptr)
            for buffer in scratch.layer_recurrent_states
        ],
        "key_ptrs": [
            None if buffer is None else int(buffer.ptr)
            for buffer in scratch.full_key_caches
        ],
        "value_ptrs": [
            None if buffer is None else int(buffer.ptr)
            for buffer in scratch.full_value_caches
        ],
    }


def _capture_compaction_group_graph(runner: Any, group: Mapping[str, Any]) -> Any:
    """Pin the graph kind the current physical-group plan would actually use."""

    request_ids = tuple(int(request_id) for request_id in group["request_ids"])
    rows = tuple(runner._rows[request_id] for request_id in request_ids)
    slots = tuple(row.slot for row in rows)
    if not slots or any(slot is None for slot in slots):
        raise RuntimeError("compaction graph group has a missing resident slot")
    concrete = tuple(slot for slot in slots if slot is not None)
    sessions = tuple(slot.session for slot in concrete)
    physical_rows = int(group["physical_rows"])
    active_slot_indices = tuple(int(index) for index in group["active_slot_indices"])
    if physical_rows == 1:
        if len(concrete) != 1 or active_slot_indices != (0,):
            raise RuntimeError("physical-c1 compaction graph has invalid membership")
        slot = concrete[0]
        existing = slot.c1_decode_graph
        if existing is not None and not bool(getattr(existing, "closed", False)):
            return existing
        capture = getattr(slot.session, "capture_decode_graph", None)
        if not callable(capture):
            raise RuntimeError("physical-c1 session cannot capture its native graph")
        graph = capture(
            position=int(slot.seq_position),
            steps_per_replay=1,
            max_replay_steps=1,
            attention_max_context_len=int(slot.seq_position) + 1,
            input_token_id=int(slot.prev_token),
        )
        slot.c1_decode_graph = graph
        return graph

    resolve_owner = getattr(runner, "_packed_execution_owner", None)
    if not callable(resolve_owner):
        raise RuntimeError("resident runner cannot resolve its packed execution owner")
    execution_owner = resolve_owner(sessions[0])
    capture = getattr(execution_owner, "capture_packed_decode_graph", None)
    if not callable(capture):
        raise RuntimeError("packed owner cannot capture its physical-group graph")
    return capture(
        [int(slot.prev_token) for slot in concrete],
        sessions=sessions,
        physical_rows=physical_rows,
        active_slot_indices=active_slot_indices,
        steps_per_replay=1,
        max_replay_steps=1,
        record_steps=1,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    logical_c = int(args.rows)
    if logical_c < 3:
        raise ValueError("rows must be at least 3 for the arbitrary-C lifecycle gate")
    cancel_slots = tuple(int(slot) for slot in args.cancel_slots)
    if len(cancel_slots) != 2 or len(set(cancel_slots)) != 2:
        raise ValueError("cancel-slots must contain two unique slots")
    if any(slot < 0 or slot >= logical_c for slot in cancel_slots):
        raise ValueError("cancel-slots must be within rows")
    original_max_tokens = int(args.original_max_tokens)
    newcomer_max_tokens = int(args.newcomer_max_tokens)
    if original_max_tokens < 5 or newcomer_max_tokens < 3:
        raise ValueError("original/newcomer max tokens must be at least 5/3")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    quality_gate = None
    if bool(args.allow_c1_arithmetic_drift):
        if args.quality_artifact is None:
            raise ValueError(
                "--allow-c1-arithmetic-drift requires --quality-artifact"
            )
        quality_gate = _load_quality_gate(
            args.quality_artifact,
            model=model,
            backend=str(args.backend),
        )

    prompt_base = int(args.prompt_token_id)
    original_prompts = tuple(
        tuple([prompt_base + row] * (int(args.prompt_length) + row))
        for row in range(logical_c)
    )
    newcomer_prompts = tuple(
        tuple([prompt_base + logical_c + row] * (int(args.prompt_length) + row + 1))
        for row in range(len(cancel_slots))
    )
    max_sequence_length = max(
        *(len(prompt) + original_max_tokens for prompt in original_prompts),
        *(len(prompt) + newcomer_max_tokens for prompt in newcomer_prompts),
    ) + 2
    compiler_version_file = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.expanduser().resolve()
    )
    if compiler_version_file is not None and not compiler_version_file.is_file():
        raise ValueError(f"compiler version file does not exist: {compiler_version_file}")

    env = {
        "HIPENGINE_MAX_ACTIVE_REQUESTS": str(logical_c),
        "HIPENGINE_PREFILL_DECODE_POLICY": "protect_ttft",
        "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
        "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
    }
    if compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_version_file)

    started = time.perf_counter()
    with _temporary_env(env):
        llm = LLM(model, backend=str(args.backend))
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(max_tokens=original_max_tokens),
            )
            adapter._loop.prefill_chunk_size = int(args.prefill_chunk_size)
            runner = adapter._runner
            if int(runner.capacity) != logical_c:
                raise RuntimeError(
                    f"expected resident capacity {logical_c}, got {runner.capacity}"
                )

            reference_session = Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                runtime=runner._shared_runner.runtime,
                shared_runner=runner._shared_runner,
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                compiler_version=(
                    None
                    if compiler_version_file is None
                    else compiler_version_file.read_text(encoding="utf-8").strip()
                ),
                require_cached_build=bool(args.require_cached_build),
            )
            try:
                original_reference_tokens: list[list[int]] = []
                original_reference_states: list[dict[int, dict[str, Any]]] = []
                for prompt in original_prompts:
                    tokens, states = _run_reference(
                        reference_session,
                        prompt,
                        max_tokens=original_max_tokens,
                    )
                    original_reference_tokens.append(tokens)
                    original_reference_states.append(states)
                    reference_session.reset()
                newcomer_reference_tokens: list[list[int]] = []
                newcomer_reference_states: list[dict[int, dict[str, Any]]] = []
                for prompt in newcomer_prompts:
                    tokens, states = _run_reference(
                        reference_session,
                        prompt,
                        max_tokens=newcomer_max_tokens,
                    )
                    newcomer_reference_tokens.append(tokens)
                    newcomer_reference_states.append(states)
                    reference_session.reset()
            finally:
                reference_session.close()

            reclaimed_states: dict[int, dict[str, Any]] = {}
            reclaimed_session_ids: dict[int, int] = {}
            original_reclaim = runner.reclaim

            def capture_reclaim(self, completed):
                request_id = int(completed.request_id)
                row = self._rows.get(request_id)
                if row is not None and row.lease is not None:
                    if row.slot is not None:
                        self._flush_row_owner(row)
                        reclaimed_states[request_id] = _capture_state(row.lease.session)
                    reclaimed_session_ids[request_id] = id(row.lease.session)
                return original_reclaim(completed)

            runner.reclaim = MethodType(capture_reclaim, runner)
            timeline: list[dict[str, Any]] = []
            token_counts: dict[int, int] = {}

            original = adapter.submit_detailed(
                _request(original_prompts, max_tokens=original_max_tokens)
            )
            original_ids = tuple(int(request_id) for request_id in original.request_ids)
            token_counts.update({request_id: 0 for request_id in original_ids})
            initial_plan = None
            ticks = 0
            while initial_plan is None:
                _poll_one(
                    adapter,
                    runner,
                    label="initial_fill",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                ticks += 1
                candidate = timeline[-1]["physical_group_plan"]
                if (
                    candidate is not None
                    and int(candidate["logical_c"]) == logical_c
                    and all(token_counts[request_id] >= 2 for request_id in original_ids)
                ):
                    initial_plan = candidate
                if ticks > original.max_ticks:
                    raise RuntimeError("initial arbitrary-C fill exceeded tick budget")

            cancelled_ids = tuple(original_ids[slot] for slot in cancel_slots)
            cancelled_session_ids = tuple(
                id(runner._rows[request_id].lease.session)
                for request_id in cancelled_ids
            )
            for slot in cancel_slots:
                adapter.cancel_submission(original, row_index=slot, reason="cancel")
            inactive_sessions = {
                id(lease.session): lease.session
                for lease in runner._available
                if id(lease.session) in cancelled_session_ids
            }
            if set(inactive_sessions) != set(cancelled_session_ids):
                raise RuntimeError("cancelled sessions did not return to the available pool")
            inactive_before_hole = {
                session_id: _capture_state(session)
                for session_id, session in inactive_sessions.items()
            }

            hole_plan = None
            while hole_plan is None:
                _poll_one(
                    adapter,
                    runner,
                    label="middle_hole_decode",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                candidate = timeline[-1]["physical_group_plan"]
                if candidate is not None and int(candidate["logical_c"]) == logical_c - 2:
                    hole_plan = candidate
            runner._flush_all_packed_owners()
            survivor_hole_states = {
                request_id: _capture_state(runner._rows[request_id].lease.session)
                for request_id in original_ids
                if request_id not in cancelled_ids
            }
            survivor_hole_counts = {
                request_id: token_counts[request_id]
                for request_id in survivor_hole_states
            }
            inactive_after_hole = {
                session_id: _capture_state(session)
                for session_id, session in inactive_sessions.items()
            }
            inactive_hole_mismatches = {
                str(session_id): _compare_one(
                    inactive_after_hole[session_id],
                    inactive_before_hole[session_id],
                )
                for session_id in cancelled_session_ids
            }

            compaction_moves: tuple[Any, ...] = ()
            moved_request_ids: tuple[int, ...] = ()
            compaction_state_hashes_before: dict[str, dict[str, Any]] = {}
            compaction_state_hashes_after: dict[str, dict[str, Any]] = {}
            compaction_state_mismatches: dict[str, list[dict[str, Any]]] = {}
            compaction_resources_before: dict[str, dict[str, Any]] = {}
            compaction_resources_after: dict[str, dict[str, Any]] = {}
            compaction_resource_mismatches: dict[str, dict[str, Any]] = {}
            compaction_graph_handles: list[dict[str, Any]] = []
            compaction_graph_invalidation_delta = 0
            compaction_graphs_closed = True
            compacted_slots: tuple[int, ...] = ()
            if bool(args.compact_after_middle_hole):
                survivor_sessions = tuple(
                    runner._rows[request_id].lease.session
                    for request_id in original_ids
                    if request_id not in cancelled_ids
                )
                # The short production lifecycle intentionally stays eager, so
                # pin real sparse physical-bucket graphs at the current state
                # before moving slots.  Compaction must retire these handles;
                # the pre/post hashes below also prove capture did not mutate
                # request-owned state or KV.
                for group in hole_plan["groups"]:
                    _capture_compaction_group_graph(runner, group)
                active_graph_handles = tuple(
                    handle
                    for handle in runner._graph_handles_for_sessions(survivor_sessions)
                    if not bool(getattr(handle, "closed", False))
                )
                survivor_resources_before = {
                    request_id: _row_resource_identity(runner._rows[request_id])
                    for request_id in original_ids
                    if request_id not in cancelled_ids
                }
                graph_invalidations_before = int(
                    runner.observability_snapshot()["graph_buckets"][
                        "invalidations_total"
                    ]
                )
                compaction_moves = tuple(adapter._loop.compact())
                moved_request_ids = tuple(
                    int(move.request_id)
                    for move in compaction_moves
                    if int(move.old_slot) != int(move.new_slot)
                )
                if not moved_request_ids:
                    raise RuntimeError("requested compaction produced no physical moves")
                compacted_slots = tuple(
                    adapter._loop.scheduler.active_batch.slot_for(request_id)
                    for request_id in original_ids
                    if request_id not in cancelled_ids
                )
                for request_id in moved_request_ids:
                    key = str(request_id)
                    before = survivor_hole_states[request_id]
                    row = runner._rows[request_id]
                    after = _capture_state(row.lease.session)
                    compaction_state_hashes_before[key] = before
                    compaction_state_hashes_after[key] = after
                    compaction_state_mismatches[key] = _compare_one(after, before)
                    compaction_resources_before[key] = survivor_resources_before[
                        request_id
                    ]
                # Resource identity is request-owned and must not change.  The
                # pre/post calls intentionally bracket no model work.
                compaction_resources_after = {
                    str(request_id): _row_resource_identity(runner._rows[request_id])
                    for request_id in moved_request_ids
                }
                compaction_resource_mismatches = {
                    key: {
                        "before": compaction_resources_before[key],
                        "after": compaction_resources_after[key],
                    }
                    for key in compaction_resources_before
                    if compaction_resources_before[key]
                    != compaction_resources_after[key]
                }
                compaction_graph_handles = [
                    {
                        "handle_id": id(handle),
                        "bucket": runner._graph_bucket_label(handle),
                        "replay_count": int(
                            getattr(handle, "replay_count", 0) or 0
                        ),
                        "closed_after_compaction": bool(
                            getattr(handle, "closed", False)
                        ),
                    }
                    for handle in active_graph_handles
                ]
                compaction_graphs_closed = bool(
                    active_graph_handles
                    and all(
                        bool(getattr(handle, "closed", False))
                        for handle in active_graph_handles
                    )
                )
                compaction_graph_invalidation_delta = int(
                    runner.observability_snapshot()["graph_buckets"][
                        "invalidations_total"
                    ]
                    - graph_invalidations_before
                )

            newcomers = adapter.submit_detailed(
                _request(newcomer_prompts, max_tokens=newcomer_max_tokens)
            )
            newcomer_ids = tuple(int(request_id) for request_id in newcomers.request_ids)
            token_counts.update({request_id: 0 for request_id in newcomer_ids})
            newcomer_slots: tuple[int, ...] | None = None
            refill_plan = None
            while refill_plan is None:
                _poll_one(
                    adapter,
                    runner,
                    label="new_admission",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                candidate = timeline[-1]["physical_group_plan"]
                if (
                    candidate is not None
                    and int(candidate["logical_c"]) == logical_c
                    and all(token_counts[request_id] >= 2 for request_id in newcomer_ids)
                ):
                    newcomer_slots = tuple(
                        adapter._loop.scheduler.active_batch.slot_for(request_id)
                        for request_id in newcomer_ids
                    )
                    refill_plan = candidate
            if newcomer_slots is None:  # pragma: no cover - guarded by refill_plan
                raise RuntimeError("newcomer physical slots were not observed")

            while not (
                adapter.generation_complete(original)
                and adapter.generation_complete(newcomers)
            ):
                _poll_one(
                    adapter,
                    runner,
                    label="drain",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                ticks += 1
                if ticks > original.max_ticks + newcomers.max_ticks:
                    raise RuntimeError("arbitrary-C lifecycle drain exceeded tick budget")

            original_outputs = tuple(adapter.take_result(original))
            newcomer_outputs = tuple(adapter.take_result(newcomers))
            original_output_tokens = [
                list(output.generated_token_ids or ()) for output in original_outputs
            ]
            newcomer_output_tokens = [
                list(output.generated_token_ids or ()) for output in newcomer_outputs
            ]
            original_finish = [
                None
                if output.finish_details is None
                else output.finish_details.to_json_dict()
                for output in original_outputs
            ]

            cancelled_mismatches = {
                str(slot): _compare_one(
                    reclaimed_states[original_ids[slot]],
                    original_reference_states[slot][
                        len(original_output_tokens[slot])
                    ],
                )
                for slot in cancel_slots
            }
            survivor_hole_mismatches = {
                str(slot): _compare_one(
                    survivor_hole_states[request_id],
                    original_reference_states[slot][
                        survivor_hole_counts[request_id]
                    ],
                )
                for slot, request_id in enumerate(original_ids)
                if request_id not in cancelled_ids
            }
            survivor_final_mismatches = {
                str(slot): _compare_one(
                    reclaimed_states[request_id],
                    original_reference_states[slot][original_max_tokens],
                )
                for slot, request_id in enumerate(original_ids)
                if request_id not in cancelled_ids
            }
            newcomer_final_mismatches = {
                str(index): _compare_one(
                    reclaimed_states[request_id],
                    newcomer_reference_states[index][newcomer_max_tokens],
                )
                for index, request_id in enumerate(newcomer_ids)
            }

            original_tokens_exact = all(
                (
                    tokens == original_reference_tokens[index]
                    if index not in cancel_slots
                    else tokens
                    == original_reference_tokens[index][: len(tokens)]
                )
                for index, tokens in enumerate(original_output_tokens)
            )
            newcomer_tokens_exact = all(
                tokens == newcomer_reference_tokens[index]
                for index, tokens in enumerate(newcomer_output_tokens)
            )
            state_exact = all(
                not mismatches
                for family in (
                    cancelled_mismatches,
                    survivor_hole_mismatches,
                    survivor_final_mismatches,
                    newcomer_final_mismatches,
                    inactive_hole_mismatches,
                )
                for mismatches in family.values()
            )
            all_plans = [
                event["physical_group_plan"]
                for event in timeline
                if event["physical_group_plan"] is not None
            ]
            declared_widths_only = all(
                int(group["physical_rows"]) in _resolve_widths()
                for plan in all_plans
                for group in plan["groups"]
            )
            no_serial_fallback = all(_all_native(plan) for plan in all_plans)
            expected_initial_masks = _expected_dense_group_masks(
                logical_c, _resolve_widths(), composition=_d2_composition(logical_c)
            )
            expected_hole_masks = _expected_hole_group_masks(
                logical_c,
                cancel_slots,
                compact=bool(args.compact_after_middle_hole),
                buckets=_resolve_widths(),
                composition=_d2_composition(logical_c),
            )
            expected_refill_masks = expected_initial_masks
            expected_newcomer_slots = (
                tuple(range(logical_c - len(cancel_slots), logical_c))
                if bool(args.compact_after_middle_hole)
                else cancel_slots
            )
            compaction_exact = bool(
                not args.compact_after_middle_hole
                or (
                    compacted_slots == tuple(range(logical_c - len(cancel_slots)))
                    and all(not rows for rows in compaction_state_mismatches.values())
                    and not compaction_resource_mismatches
                    and compaction_graphs_closed
                    and compaction_graph_invalidation_delta
                    == len(compaction_graph_handles)
                )
            )
            observability = runner.observability_snapshot()
            routes = observability["routes"]
            final_active = tuple(runner.active_request_ids)
            final_available = int(runner.available_session_count)
            scheduler_active = int(adapter._loop.active_count)
            cancelled_finish_ok = all(
                original_finish[slot] == {"reason": "cancelled", "cancelled": True}
                for slot in cancel_slots
            )
            newcomer_reclaimed_session_ids = tuple(
                reclaimed_session_ids[request_id] for request_id in newcomer_ids
            )
            session_reused = set(newcomer_reclaimed_session_ids) == set(
                cancelled_session_ids
            )
            state_kv_accepted = _state_kv_accepted(
                bit_exact=state_exact,
                allow_c1_arithmetic_drift=bool(args.allow_c1_arithmetic_drift),
            )
            provenance = collect_artifact_provenance(
                repo_root=_REPO_ROOT,
                configured_backend=str(args.backend),
                resolved_backend=str(runner._shared_runner.backend),
                target_arch=str(runner._shared_runner.target_arch),
                model_path=model,
                quant="gguf_q4_k_m",
                kv_dtype="bf16",
                command=[sys.executable, *sys.argv],
                environment={
                    "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                    "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                    "HIPENGINE_GGUF_GDN_PREFILL_MODE": os.environ.get(
                        "HIPENGINE_GGUF_GDN_PREFILL_MODE"
                    ),
                    "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS": os.environ.get(
                        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS"
                    ),
                    "HIPENGINE_GGUF_AR_D2_COST_ARTIFACT": os.environ.get(
                        "HIPENGINE_GGUF_AR_D2_COST_ARTIFACT"
                    ),
                    "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL": os.environ.get(
                        "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"
                    ),
                    "HIPENGINE_GGUF_ROUTER_F32W_COOP": os.environ.get(
                        "HIPENGINE_GGUF_ROUTER_F32W_COOP"
                    ),
                },
                build_profile="gguf_arbitrary_c_lifecycle",
                timing_protocol="none_correctness_only_v1",
                warmups=0,
                repetitions=1,
                profiler={"enabled": False, "kind": None, "command": None},
            )
            passed = bool(
                original_tokens_exact
                and newcomer_tokens_exact
                and state_kv_accepted
                and cancelled_finish_ok
                and newcomer_slots == expected_newcomer_slots
                and session_reused
                and compaction_exact
                and _group_masks(initial_plan) == expected_initial_masks
                and _group_masks(hole_plan) == expected_hole_masks
                and _group_masks(refill_plan) == expected_refill_masks
                and declared_widths_only
                and no_serial_fallback
                and routes["fallback_reasons"] == {}
                and final_active == ()
                and final_available == logical_c
                and scheduler_active == 0
            )
            return {
                "schema": 3,
                "kind": "gguf_arbitrary_c_lifecycle",
                "status": "passed" if passed else "failed",
                "passed": passed,
                "performance_claim": False,
                "provenance": provenance,
                "model": str(model),
                "backend": str(args.backend),
                "target_arch": str(runner._shared_runner.target_arch),
                "shape": {
                    "logical_c": logical_c,
                    "cancel_slots": list(cancel_slots),
                    "newcomer_slots": list(newcomer_slots),
                    "compact_after_middle_hole": bool(
                        args.compact_after_middle_hole
                    ),
                    "prompt_lengths": [len(prompt) for prompt in original_prompts],
                    "newcomer_prompt_lengths": [len(prompt) for prompt in newcomer_prompts],
                    "original_max_tokens": original_max_tokens,
                    "newcomer_max_tokens": newcomer_max_tokens,
                    "prefill_chunk_size": int(args.prefill_chunk_size),
                },
                "initial_plan": initial_plan,
                "middle_hole_plan": hole_plan,
                "new_admission_plan": refill_plan,
                "all_decode_plans": all_plans,
                "declared_widths_only": declared_widths_only,
                "no_serial_fallback": no_serial_fallback,
                "original_request_ids": list(original_ids),
                "cancelled_request_ids": list(cancelled_ids),
                "newcomer_request_ids": list(newcomer_ids),
                "original_reference_tokens": original_reference_tokens,
                "original_output_tokens": original_output_tokens,
                "newcomer_reference_tokens": newcomer_reference_tokens,
                "newcomer_output_tokens": newcomer_output_tokens,
                "original_tokens_exact": original_tokens_exact,
                "newcomer_tokens_exact": newcomer_tokens_exact,
                "state_kv_exact": state_exact,
                "state_kv_c1_bit_exact": state_exact,
                "state_kv_accepted": state_kv_accepted,
                "allow_c1_arithmetic_drift": bool(args.allow_c1_arithmetic_drift),
                "external_numerical_quality_gate": quality_gate,
                "state_kv_mismatches": {
                    "cancelled": cancelled_mismatches,
                    "survivors_middle_hole": survivor_hole_mismatches,
                    "survivors_final": survivor_final_mismatches,
                    "newcomers_final": newcomer_final_mismatches,
                    "inactive_sessions_during_middle_hole": inactive_hole_mismatches,
                },
                "finish_details": original_finish,
                "cancelled_session_ids": list(cancelled_session_ids),
                "newcomer_reclaimed_session_ids": list(
                    newcomer_reclaimed_session_ids
                ),
                "cancelled_sessions_reused_by_newcomers": session_reused,
                "compaction": {
                    "enabled": bool(args.compact_after_middle_hole),
                    "passed": compaction_exact,
                    "moves": [
                        {
                            "request_id": int(move.request_id),
                            "old_slot": int(move.old_slot),
                            "new_slot": int(move.new_slot),
                        }
                        for move in compaction_moves
                    ],
                    "moved_request_ids": list(moved_request_ids),
                    "compacted_slots": list(compacted_slots),
                    "state_kv_hashes_before": compaction_state_hashes_before,
                    "state_kv_hashes_after": compaction_state_hashes_after,
                    "state_kv_mismatches": compaction_state_mismatches,
                    "resource_identity_before": compaction_resources_before,
                    "resource_identity_after": compaction_resources_after,
                    "resource_identity_mismatches": compaction_resource_mismatches,
                    "graph_handles_before": compaction_graph_handles,
                    "graph_handles_closed": compaction_graphs_closed,
                    "graph_invalidation_delta": compaction_graph_invalidation_delta,
                },
                "graph_buckets": observability["graph_buckets"],
                "routes": routes,
                "timeline": timeline,
                "final_active_request_ids": list(final_active),
                "final_available_sessions": final_available,
                "final_scheduler_active": scheduler_active,
                "elapsed_seconds": time.perf_counter() - started,
                "notes": [
                    (
                        "Optional compaction packs survivors after the middle-hole transition and preserves request-owned state/KV resources."
                        if args.compact_after_middle_hole
                        else "The gate preserves scheduler slot identity; no physical compaction is performed."
                    ),
                    "Every decode group must use only the active declared physical widths "
                    f"{list(_resolve_widths())}.",
                    "Tokens, Conv/GDN state, and all live BF16 KV bytes are compared with c1 checkpoints; arithmetic drift remains reported even when an external numerical gate makes byte identity non-binding.",
                    "Same-run state/KV preservation across compaction, ownership, routes, masks, and graph invalidation remain hard requirements.",
                    "The CLI additionally binds tracked allocator recovery after model teardown.",
                ],
            }
        finally:
            llm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--cancel-slots", type=int, nargs=2, default=(2, 10))
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--original-max-tokens", type=int, default=5)
    parser.add_argument("--newcomer-max-tokens", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument(
        "--compact-after-middle-hole",
        action="store_true",
        help="compact survivor slots after the middle-hole transition",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--allow-c1-arithmetic-drift",
        action="store_true",
        help="report but do not bind cN-vs-c1 state/KV byte differences",
    )
    parser.add_argument(
        "--quality-artifact",
        type=Path,
        help="matching passed distributional gate required with arithmetic drift",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tracked_before = dict(memory_stats())
    payload = run(args)
    tracked_after = dict(memory_stats())
    memory = _tracked_memory_recovery(tracked_before, tracked_after)
    payload["memory"] = memory
    payload["passed"] = bool(payload["passed"] and memory["passed"])
    payload["status"] = "passed" if payload["passed"] else "failed"
    command_args = list(sys.argv[1:] if argv is None else argv)
    payload["command"] = shlex.join(
        [sys.executable, "scripts/gguf_arbitrary_c_lifecycle.py", *command_args]
    )
    payload["environment"] = {
        key: os.environ.get(key)
        for key in (
            "HIPENGINE_HIP_ARCH",
            "HIP_VISIBLE_DEVICES",
            "HIPENGINE_COMPILER_VERSION_FILE",
            "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS",
            "HIPENGINE_GGUF_AR_D2_COST_ARTIFACT",
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
