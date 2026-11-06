"""Legacy native-row GGUF batch diagnostic, separate from resident scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hipengine.dispatch import WorkKind
from hipengine.generation.batch_scheduler import GeneratedToken, ResidentBatchScheduler
from hipengine.generation.deadline import raise_if_generation_deadline_expired


@dataclass(frozen=True)
class GGUFNativeBatchRun:
    generated_ids: dict[int, list[int]]
    native_decode_steps: int
    execution_paths: dict[str, str]
    scheduling: dict[str, Any]


def generate_native_greedy_batch(
    session: Any,
    prompt_rows_by_request: dict[int, list[int]],
    request: Any,
    *,
    capacity: int,
    tokenizer: Any,
) -> GGUFNativeBatchRun:
    """Continuously admit and compact greedy rows around native c-aware decode."""
    if len(prompt_rows_by_request) <= 1:
        raise ValueError("native GGUF greedy batch requires at least two prompts")
    if int(capacity) < 2 or int(capacity) > 8:
        raise ValueError("native GGUF greedy batch capacity must be within [2, 8]")

    scheduler = ResidentBatchScheduler(capacity=int(capacity), context_bucket_size=256)
    for request_id in sorted(prompt_rows_by_request):
        scheduler.submit(
            prompt_rows_by_request[request_id],
            max_new_tokens=int(request.max_tokens),
            request_id=int(request_id),
            sampling_row_index=int(request_id),
        )

    native_decode_steps = 0
    single_row_tail_steps = 0
    eager_native_ready = False
    execution_paths: dict[str, str] = {}
    graph_objects: list[object] = []
    graph_bucket_labels: list[str] = []
    active_c_histogram: dict[int, int] = {}
    admission_history: list[dict[str, int]] = []
    admission_waves = 0
    reclaim_count = 0
    compaction_events = 0
    compacted_slot_moves = 0
    mixed_prefill_decode_admissions = 0
    peak_active_rows = 0

    config = getattr(getattr(getattr(session, "runner", None), "weights", None), "config", None)
    experts_per_token = int(getattr(config, "expert_used_count", 0) or 0)
    kv_storage = getattr(getattr(session, "kv_storage_dtype", None), "value", "bf16")
    resident_max_sequence_length = int(
        getattr(
            getattr(session, "target_layout", None),
            "max_sequence_length",
            max(len(tokens) for tokens in prompt_rows_by_request.values())
            + int(request.max_tokens),
        )
    )

    def compact_after_reclaim(completed_count: int) -> None:
        nonlocal compaction_events, compacted_slot_moves
        if completed_count <= 0:
            return
        moves = scheduler.compact()
        if scheduler.active_count or scheduler.pending_count:
            source_slots = tuple(int(move.old_slot) for move in moves)
            session.compact_target_slots(source_slots)
            compaction_events += 1
            compacted_slot_moves += sum(
                int(move.old_slot) != int(move.new_slot) for move in moves
            )

    def admit_available() -> None:
        nonlocal admission_waves, reclaim_count, mixed_prefill_decode_admissions, peak_active_rows
        while scheduler.pending_count and scheduler.active_count < int(capacity):
            admitted = scheduler.admit_pending()
            if not admitted:
                break
            admission_waves += 1
            if native_decode_steps or single_row_tail_steps:
                mixed_prefill_decode_admissions += len(admitted)
            peak_active_rows = max(peak_active_rows, scheduler.active_count)
            completed_in_wave = 0
            for request_id in admitted:
                work = scheduler.next_prefill_work(
                    chunk_size=resident_max_sequence_length,
                )
                if work is None or work.request_ids != (int(request_id),):
                    raise RuntimeError("GGUF scheduler prefill order diverged from admission order")
                slot = scheduler.active_batch.slot_for(request_id)
                admission_history.append(
                    {
                        "request_id": int(request_id),
                        "slot": int(slot),
                        "wave": int(admission_waves),
                    }
                )
                raise_if_generation_deadline_expired(request)
                result = session.prefill_slot(
                    work.token_rows[0],
                    slot=slot,
                    return_logits=False,
                )
                first_token = int(result.token_id)
                finished = (
                    not request.ignore_eos
                    and first_token == int(tokenizer.eos_token_id)
                )
                completed = scheduler.record_generated(
                    (GeneratedToken(int(request_id), first_token, finished=finished),)
                )
                completed_in_wave += len(completed)
                reclaim_count += len(completed)
            compact_after_reclaim(completed_in_wave)

    def graph_for(key):
        def create_graph(_key):
            current_positions = tuple(
                int(position)
                for position in getattr(
                    session,
                    "row_positions",
                    (max(0, int(_key.context_bucket) - 1),) * int(_key.active_c),
                )[: int(_key.active_c)]
            )
            required_bound = max(current_positions) + 1
            max_sequence_length = resident_max_sequence_length
            context_bound = min(
                max_sequence_length,
                max(required_bound, int(_key.context_bucket)),
            )
            graph = session.capture_native_rows_graph(
                rows=int(_key.active_c),
                max_context_len=context_bound,
            )
            graph_objects.append(graph)
            return graph

        return scheduler.graph_buckets.get_or_create(
            key,
            create_graph,
            miss_reason="gguf_native_shape_absent",
        )

    try:
        while scheduler.pending_count or scheduler.active_count:
            admit_available()
            if scheduler.active_count == 0:
                continue
            work = scheduler.next_decode_work(
                top_k=experts_per_token,
                experts_per_token=experts_per_token,
                replay_steps=1,
                kv_storage_dtype=str(kv_storage),
                layer_plan="qwen35_gguf_native",
            )
            if work is None:
                raise RuntimeError("GGUF scheduler has active rows but no decode work")
            request_ids = tuple(int(request_id) for request_id in work.request_ids)
            current_tokens = tuple(
                int(scheduler.active_batch.requests[request_id].generated_tokens[-1])
                for request_id in request_ids
            )
            active_rows = len(request_ids)
            active_c_histogram[active_rows] = active_c_histogram.get(active_rows, 0) + 1
            key = scheduler.shape_key(
                mode=WorkKind.DECODE,
                top_k=experts_per_token,
                experts_per_token=experts_per_token,
                replay_steps=1,
                kv_storage_dtype=str(kv_storage),
                layer_plan="qwen35_gguf_native",
            )
            bucket_label = (
                f"decode:c={key.active_c}:ctx={key.context_bucket}:"
                f"mask={''.join('1' if active else '0' for active in key.active_mask)}:"
                f"top_k={key.top_k}:experts={key.experts_per_token}"
            )
            if bucket_label not in graph_bucket_labels:
                graph_bucket_labels.append(bucket_label)

            raise_if_generation_deadline_expired(request)
            if active_rows == 1:
                step = session.step(current_tokens[0], return_logits=False)
                next_tokens = (int(step.token_id),)
                single_row_tail_steps += 1
                execution_paths["single_row_tail"] = "resident_slot0_c1"
            elif not eager_native_ready or getattr(session, "host_token_embedding_enabled", False):
                step = session.step_rows_native(current_tokens, return_logits=False)
                next_tokens = tuple(int(token) for token in step.token_ids)
                execution_paths.update(dict(step.execution_paths))
                native_decode_steps += 1
                eager_native_ready = True
            else:
                graph = graph_for(key)
                step = graph.step(current_tokens)
                scheduler.graph_buckets.record_replay_kernel_hit()
                next_tokens = tuple(int(token) for token in step.token_ids)
                execution_paths.update(dict(step.execution_paths))
                native_decode_steps += 1

            completed = scheduler.record_generated(
                tuple(
                    GeneratedToken(
                        request_id,
                        token_id,
                        finished=(
                            not request.ignore_eos
                            and token_id == int(tokenizer.eos_token_id)
                        ),
                    )
                    for request_id, token_id in zip(request_ids, next_tokens, strict=True)
                )
            )
            reclaim_count += len(completed)
            compact_after_reclaim(len(completed))
            peak_active_rows = max(peak_active_rows, scheduler.active_count)
    finally:
        for graph in reversed(graph_objects):
            close = getattr(graph, "close", None)
            if callable(close):
                close()
                continue
            exit_graph = getattr(graph, "__exit__", None)
            if callable(exit_graph):
                exit_graph(None, None, None)

    generated = {
        request_id: list(scheduler.completed[request_id].generated_tokens)
        for request_id in sorted(prompt_rows_by_request)
    }
    graph_stats = scheduler.graph_buckets.stats.to_json_dict()
    scheduling = {
        "continuous_batching": True,
        "capacity": int(capacity),
        "admission_count": len(admission_history),
        "admission_waves": int(admission_waves),
        "admission_history": admission_history,
        "reclaim_count": int(reclaim_count),
        "compaction_events": int(compaction_events),
        "compacted_slot_moves": int(compacted_slot_moves),
        "mixed_prefill_decode_admissions": int(mixed_prefill_decode_admissions),
        "peak_active_rows": int(peak_active_rows),
        "active_c_histogram": {
            str(active_rows): int(steps)
            for active_rows, steps in sorted(active_c_histogram.items())
        },
        "graph_bucket_keys": graph_bucket_labels,
        "graph_bucket_stats": graph_stats,
        "single_row_tail_steps": int(single_row_tail_steps),
        "stable_request_ids": sorted(prompt_rows_by_request),
        "request_observability": {
            str(request_id): scheduler.completed[request_id].observability.to_json_dict()
            for request_id in sorted(prompt_rows_by_request)
        },
        "final_request_to_slot": scheduler.active_batch.request_to_slot,
        "serial_decode_fallback": False,
    }
    return GGUFNativeBatchRun(
        generated_ids=generated,
        native_decode_steps=native_decode_steps,
        execution_paths=execution_paths,
        scheduling=scheduling,
    )
