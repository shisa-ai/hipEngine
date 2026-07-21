"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Iterator

from hipengine.dispatch import WorkKind
from hipengine.generation.batch_scheduler import GeneratedToken, ResidentBatchScheduler
from hipengine.generation.constraints import token_sequence_state_for_tokens
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
    TokenLogprob,
    register_text_generator,
)
from hipengine.generation.sampling import (
    RowSamplingState,
    SamplingMode,
    plan_sampler,
    row_seed_for_index,
    select_token,
    thinking_budget_state_from_params,
)
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


@dataclass(frozen=True)
class _GGUFNativeBatchRun:
    generated_ids: dict[int, list[int]]
    native_decode_steps: int
    execution_paths: dict[str, str]
    scheduling: dict[str, Any]


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF greedy generator over a persistent resident session."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    bulk_prefill_attention_mode: str = "bulk"
    prefill_quant: str | None = None
    prefill_attn_aotriton_min_tokens: int | None = None
    native_batch_decode: bool = False
    native_batch_capacity: int = 8
    tokenizer: Qwen35GGUFTokenizer = field(init=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)
    supports_stream_logprobs: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.bulk_prefill_attention_mode not in {"bulk", "native"}:
            raise ValueError("bulk_prefill_attention_mode must be 'bulk' or 'native'")
        if (
            self.prefill_attn_aotriton_min_tokens is not None
            and int(self.prefill_attn_aotriton_min_tokens) < 0
        ):
            raise ValueError("prefill_attn_aotriton_min_tokens must be non-negative")
        if int(self.native_batch_capacity) < 2 or int(self.native_batch_capacity) > 8:
            raise ValueError("native_batch_capacity must be within [2, 8]")
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

    def _configure_session(self, session: Qwen35GGUFResidentSession) -> None:
        # Prefill correctness policy is selected by the generator registry
        # factory, not by a quant branch in runtime dispatch.
        session.default_bulk_attention_mode = getattr(
            self,
            "bulk_prefill_attention_mode",
            "bulk",
        )
        if self.prefill_quant is not None:
            session.select_prefill_quant(self.prefill_quant)
        if self.prefill_attn_aotriton_min_tokens is not None:
            session.prefill_config = replace(
                session.prefill_config or PrefillConfig(),
                attn_aotriton_min_tokens=int(self.prefill_attn_aotriton_min_tokens),
            )

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(self, request: GenerationRequest) -> list[str]:
        outputs = self.generate_detailed(request)
        return [output.text for output in outputs]

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield chunk.text

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        self.last_batch_generation = None
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        if request.max_tokens == 0:
            return
        prompt_ids = self.tokenizer.encode(request.prompts[0])
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("GGUF prompt tokenization produced no token IDs")
        plan = _gguf_sampler_plan(request)
        with Qwen35GGUFResidentSession(self.model_path) as session:
            self._configure_session(session)
            if plan.mode is SamplingMode.GREEDY_FAST:
                yield from self._stream_greedy(session, prompt_ids, request)
                return
            yield from self._stream_sampled(
                session,
                prompt_ids,
                request,
                row_index=0,
            )

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        plan = _gguf_sampler_plan(request)
        if request.max_tokens == 0:
            prompt_rows_by_request = {
                index: self.tokenizer.encode(prompt)
                for index, prompt in enumerate(request.prompts)
            }
            self.last_generation_outputs = tuple(
                GenerationOutput(
                    text="",
                    finish_details=_gguf_finish_details((), self.tokenizer, request),
                    telemetry=_gguf_telemetry(
                        prompt_rows_by_request[index],
                        (),
                        request,
                        row_index=index,
                    ),
                )
                for index, prompt in enumerate(request.prompts)
            )
            self.last_batch_generation = _gguf_last_batch_generation(
                self.tokenizer,
                request,
                plan,
                prompt_rows_by_request,
                {index: [] for index in prompt_rows_by_request},
                {index: [] for index in prompt_rows_by_request},
                outputs=self.last_generation_outputs,
            )
            return list(self.last_generation_outputs)
        outputs: list[GenerationOutput] = []
        prompt_rows_by_request: dict[int, list[int]] = {}
        generated_ids_by_request: dict[int, list[int]] = {}
        token_logprobs_by_request: dict[int, list[TokenLogprob]] = {}
        if (
            plan.mode is SamplingMode.GREEDY_FAST
            and len(request.prompts) > 1
            and bool(getattr(self, "native_batch_decode", False))
        ):
            prompt_rows_by_request = {
                row_index: self.tokenizer.encode(prompt)
                for row_index, prompt in enumerate(request.prompts)
            }
            if any(not prompt_ids for prompt_ids in prompt_rows_by_request.values()):
                raise ValueError("GGUF prompt tokenization produced no token IDs")
            max_sequence_length = max(
                256,
                max(len(prompt_ids) for prompt_ids in prompt_rows_by_request.values())
                + int(request.max_tokens),
            )
            native_capacity = min(
                int(getattr(self, "native_batch_capacity", 8)),
                len(prompt_rows_by_request),
            )
            with Qwen35GGUFResidentSession(
                self.model_path,
                max_sequence_length=max_sequence_length,
                max_batch_size=native_capacity,
            ) as session:
                self._configure_session(session)
                native_run = self._generate_greedy_batch(
                    session,
                    prompt_rows_by_request,
                    request,
                    capacity=native_capacity,
                )
                generated_ids_by_request = native_run.generated_ids
            outputs = [
                GenerationOutput(
                    text=self.tokenizer.decode(generated_ids_by_request[row_index]),
                    finish_details=_gguf_finish_details(
                        generated_ids_by_request[row_index],
                        self.tokenizer,
                        request,
                    ),
                    telemetry=_gguf_telemetry(
                        prompt_rows_by_request[row_index],
                        generated_ids_by_request[row_index],
                        request,
                        row_index=row_index,
                        request_id=str(row_index),
                        phase="answer",
                        execution_path="gguf_native_continuous_decode",
                        native_compact_prefill=False,
                        native_caware_decode=True,
                        serial_decode_fallback=False,
                        native_sampler_rows=True,
                    ),
                )
                for row_index in range(len(prompt_rows_by_request))
            ]
            self.last_generation_outputs = tuple(outputs)
            self.last_batch_generation = _gguf_last_batch_generation(
                self.tokenizer,
                request,
                plan,
                prompt_rows_by_request,
                generated_ids_by_request,
                token_logprobs_by_request,
                outputs=self.last_generation_outputs,
                native_batch=True,
                native_decode_steps=native_run.native_decode_steps,
                execution_paths=native_run.execution_paths,
                scheduling=native_run.scheduling,
            )
            return outputs

        with Qwen35GGUFResidentSession(self.model_path) as session:
            self._configure_session(session)
            for row_index, prompt in enumerate(request.prompts):
                raise_if_generation_deadline_expired(request)
                prompt_ids = self.tokenizer.encode(prompt)
                prompt_rows_by_request[row_index] = prompt_ids
                raise_if_generation_deadline_expired(request)
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                if plan.mode is SamplingMode.GREEDY_FAST:
                    generated_ids = self._generate_greedy(session, prompt_ids, request)
                    generated_ids_by_request[row_index] = list(generated_ids)
                    finish_details = _gguf_finish_details(generated_ids, self.tokenizer, request)
                    outputs.append(
                        GenerationOutput(
                            text=self.tokenizer.decode(generated_ids),
                            finish_details=finish_details,
                            telemetry=_gguf_telemetry(
                                prompt_ids,
                                generated_ids,
                                request,
                                row_index=row_index,
                            ),
                        )
                    )
                else:
                    output = self._generate_sampled(
                        session,
                        prompt_ids,
                        request,
                        row_index=row_index,
                    )
                    outputs.append(output)
                    token_logprobs_by_request[row_index] = list(output.token_logprobs)
                    generated_ids_by_request[row_index] = [
                        int(token.token_id) for token in output.token_logprobs
                    ]
        self.last_generation_outputs = tuple(outputs)
        self.last_batch_generation = _gguf_last_batch_generation(
            self.tokenizer,
            request,
            plan,
            prompt_rows_by_request,
            generated_ids_by_request,
            token_logprobs_by_request,
            outputs=self.last_generation_outputs,
        )
        return outputs

    def _generate_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
    ) -> list[int]:
        generated_ids: list[int] = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=False)
        raise_if_generation_deadline_expired(request)
        generated_ids.append(int(result.token_id))
        if request.ignore_eos or int(result.token_id) != self.tokenizer.eos_token_id:
            remaining = request.max_tokens - 1
            if remaining > 0:
                if _session_uses_host_routed_decode(session):
                    for _ in range(remaining):
                        raise_if_generation_deadline_expired(request)
                        step = session.step(generated_ids[-1], return_logits=False)
                        raise_if_generation_deadline_expired(request)
                        generated_ids.append(int(step.token_id))
                        if (
                            not request.ignore_eos
                            and int(step.token_id) == self.tokenizer.eos_token_id
                        ):
                            break
                else:
                    with session.capture_decode_graph(
                        position=len(prompt_ids),
                        steps_per_replay=1,
                        max_replay_steps=remaining,
                        record_steps=remaining,
                    ) as graph:
                        raise_if_generation_deadline_expired(request)
                        graph.replay(remaining)
                        raise_if_generation_deadline_expired(request)
                        for token_id in graph.read_generated_token_ids(remaining):
                            raise_if_generation_deadline_expired(request)
                            generated_ids.append(int(token_id))
                            if (
                                not request.ignore_eos
                                and int(token_id) == self.tokenizer.eos_token_id
                            ):
                                break
        return generated_ids

    def _generate_greedy_batch(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_rows_by_request: dict[int, list[int]],
        request: GenerationRequest,
        *,
        capacity: int,
    ) -> _GGUFNativeBatchRun:
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
                        and first_token == int(self.tokenizer.eos_token_id)
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
                                and token_id == int(self.tokenizer.eos_token_id)
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
        return _GGUFNativeBatchRun(
            generated_ids=generated,
            native_decode_steps=native_decode_steps,
            execution_paths=execution_paths,
            scheduling=scheduling,
        )

    def _generate_sampled(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
    ) -> GenerationOutput:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        state = _gguf_row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        samples = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        sample = _select_from_gguf_logits(result, sampling_request, state)
        samples.append(sample)
        generated_ids = [int(sample.token_id)]
        _gguf_queue_json_object_close_if_needed(
            state,
            self.tokenizer,
            _gguf_token_text(self.tokenizer, sample),
            remaining_tokens=request.max_tokens - len(generated_ids),
        )
        if _gguf_finished(generated_ids, self.tokenizer, request):
            return _gguf_generation_output(
                self.tokenizer,
                samples,
                finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request, state),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    request,
                    row_index=row_index,
                    sampling_state=state,
                    forced_sample=sample,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                ),
            )
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=True)
            raise_if_generation_deadline_expired(request)
            step_full_vocab_logits_d2h, step_logits_d2h_bytes = _gguf_logits_d2h_metadata(step)
            if step_full_vocab_logits_d2h is not None:
                full_vocab_logits_d2h = step_full_vocab_logits_d2h
                logits_d2h_bytes = step_logits_d2h_bytes
            sample = _select_from_gguf_logits(step, sampling_request, state)
            samples.append(sample)
            generated_ids.append(int(sample.token_id))
            _gguf_queue_json_object_close_if_needed(
                state,
                self.tokenizer,
                _gguf_token_text(self.tokenizer, sample),
                remaining_tokens=request.max_tokens - len(generated_ids),
            )
            if _gguf_finished(generated_ids, self.tokenizer, request):
                break
        return _gguf_generation_output(
            self.tokenizer,
            samples,
            finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request, state),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                request,
                row_index=row_index,
                sampling_state=state,
                forced_sample=samples[-1] if samples else None,
                full_vocab_logits_d2h=full_vocab_logits_d2h,
                logits_d2h_bytes=logits_d2h_bytes,
            ),
        )

    def _stream_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        generated_ids: list[int] = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=False)
        raise_if_generation_deadline_expired(request)
        generated_ids.append(int(result.token_id))
        finished = _gguf_finished(generated_ids, self.tokenizer, request)
        yield GenerationStreamChunk(
            self.tokenizer.decode([generated_ids[-1]]),
            finish_details=(
                _gguf_finish_details(generated_ids, self.tokenizer, request)
                if finished or len(generated_ids) >= request.max_tokens
                else None
            ),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                request,
                row_index=0,
                phase="answer",
            ),
        )
        if finished:
            return
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=False)
            raise_if_generation_deadline_expired(request)
            generated_ids.append(int(step.token_id))
            finished = _gguf_finished(generated_ids, self.tokenizer, request)
            yield GenerationStreamChunk(
                self.tokenizer.decode([generated_ids[-1]]),
                finish_details=(
                    _gguf_finish_details(generated_ids, self.tokenizer, request)
                    if finished or len(generated_ids) >= request.max_tokens
                    else None
                ),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    request,
                    row_index=0,
                    phase="answer",
                ),
            )
            if finished:
                return

    def _stream_sampled(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
    ) -> Iterator[GenerationStreamChunk]:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        state = _gguf_row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        generated_ids: list[int] = []
        live_phase = None if state.thinking_budget is not None else "answer"
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        sample = _select_from_gguf_logits(result, sampling_request, state)
        generated_ids.append(int(sample.token_id))
        _gguf_queue_json_object_close_if_needed(
            state,
            self.tokenizer,
            _gguf_token_text(self.tokenizer, sample),
            remaining_tokens=request.max_tokens - len(generated_ids),
        )
        finished = _gguf_finished(generated_ids, self.tokenizer, sampling_request)
        yield GenerationStreamChunk(
            self.tokenizer.decode([generated_ids[-1]]),
            token_logprobs=_gguf_stream_token_logprobs(self.tokenizer, sample, sampling_request),
            finish_details=(
                _gguf_finish_details(generated_ids, self.tokenizer, sampling_request, state)
                if finished or len(generated_ids) >= sampling_request.max_tokens
                else None
            ),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                sampling_request,
                row_index=row_index,
                sampling_state=state,
                phase=live_phase,
                forced_sample=sample,
                full_vocab_logits_d2h=full_vocab_logits_d2h,
                logits_d2h_bytes=logits_d2h_bytes,
            ),
        )
        if finished:
            return
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=True)
            raise_if_generation_deadline_expired(request)
            full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(step)
            sample = _select_from_gguf_logits(step, sampling_request, state)
            generated_ids.append(int(sample.token_id))
            _gguf_queue_json_object_close_if_needed(
                state,
                self.tokenizer,
                _gguf_token_text(self.tokenizer, sample),
                remaining_tokens=request.max_tokens - len(generated_ids),
            )
            finished = _gguf_finished(generated_ids, self.tokenizer, sampling_request)
            yield GenerationStreamChunk(
                self.tokenizer.decode([generated_ids[-1]]),
                token_logprobs=_gguf_stream_token_logprobs(self.tokenizer, sample, sampling_request),
                finish_details=(
                    _gguf_finish_details(generated_ids, self.tokenizer, sampling_request, state)
                    if finished or len(generated_ids) >= sampling_request.max_tokens
                    else None
                ),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    sampling_request,
                    row_index=row_index,
                    sampling_state=state,
                    phase=live_phase,
                    forced_sample=sample,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                ),
            )
            if finished:
                return


def _select_from_gguf_logits(
    result: Any,
    request: GenerationRequest,
    state: RowSamplingState,
):
    logits = getattr(result, "logits", None)
    if logits is None:
        raise RuntimeError("GGUF sampled generation requires logits from the resident session")
    return select_token(logits.reshape(-1), request, state)


def _gguf_logits_d2h_metadata(result: Any) -> tuple[bool | None, int | None]:
    logits = getattr(result, "logits", None)
    if logits is None:
        return None, None
    size = getattr(logits, "size", None)
    itemsize = getattr(getattr(logits, "dtype", None), "itemsize", None)
    try:
        if int(size) > 0 and int(itemsize) > 0:
            return True, int(size) * int(itemsize)
    except (TypeError, ValueError):
        pass
    shape = getattr(logits, "shape", None)
    if shape:
        try:
            vocab_size = int(shape[-1])
        except (TypeError, ValueError):
            return True, None
        if vocab_size > 0:
            return True, vocab_size * 4
    return True, None


def _request_with_tokenizer_eos(
    request: GenerationRequest,
    tokenizer: Qwen35GGUFTokenizer,
) -> GenerationRequest:
    if request.eos_token_id is not None:
        return request
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return request
    return replace(request, eos_token_id=int(eos_token_id))


def _gguf_row_sampling_state(
    request: GenerationRequest,
    prompt_ids: list[int],
    *,
    row_index: int,
) -> RowSamplingState:
    return RowSamplingState(
        prompt_tokens=tuple(int(token) for token in prompt_ids),
        seed=row_seed_for_index(request, row_index),
        row_index=row_index,
        stop_token_sequences=request.stop_token_sequences,
        forced_tokens_pending=request.forced_tokens_pending,
        forced_token_reason=request.forced_token_reason,
        post_thinking_forced_tokens_pending=request.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=request.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=request.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=request.force_sequence_completion_reason,
        json_object_close_forcing=request.json_object_close_forcing,
        thinking_budget=thinking_budget_state_from_params(request),
    )


def _gguf_generation_output(
    tokenizer: Qwen35GGUFTokenizer,
    samples,
    *,
    finish_details: FinishDetails,
    telemetry: GenerationTelemetry | None = None,
) -> GenerationOutput:
    token_logprobs = tuple(_gguf_token_logprob(tokenizer, sample) for sample in samples)
    return GenerationOutput(
        text="".join(token.token_text for token in token_logprobs),
        token_logprobs=token_logprobs,
        finish_details=finish_details,
        telemetry=telemetry,
    )


def _gguf_stream_token_logprobs(
    tokenizer: Qwen35GGUFTokenizer,
    sample: Any,
    request: GenerationRequest,
) -> tuple[TokenLogprob, ...]:
    if not request.logprobs and int(request.top_logprobs) <= 0:
        return ()
    return (_gguf_token_logprob(tokenizer, sample),)


def _gguf_token_logprob(tokenizer: Qwen35GGUFTokenizer, sample: Any) -> TokenLogprob:
    return TokenLogprob(
        token_id=sample.token_id,
        token_text=_gguf_token_text(tokenizer, sample),
        logprob=sample.logprob,
        top_logprobs=tuple(
            (token_id, tokenizer.decode([int(token_id)]), logprob)
            for token_id, logprob in sample.top_logprobs
        ),
    )


def _gguf_token_text(tokenizer: Qwen35GGUFTokenizer, sample: Any) -> str:
    token_text = getattr(sample, "token_text", None)
    if token_text is not None:
        return str(token_text)
    return tokenizer.decode([int(sample.token_id)])


def _gguf_last_batch_generation(
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    plan: Any,
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids_by_request: dict[int, list[int]],
    token_logprobs_by_request: dict[int, list[TokenLogprob]],
    *,
    outputs: tuple[GenerationOutput, ...],
    native_batch: bool = False,
    native_decode_steps: int = 0,
    execution_paths: dict[str, str] | None = None,
    scheduling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_ids = tuple(range(len(outputs)))
    path = "gguf_native_continuous_decode" if native_batch else _gguf_execution_path(plan)
    prompt_lengths = [len(prompt_rows_by_request.get(request_id, ())) for request_id in request_ids]
    decode_steps = max((len(generated_ids_by_request.get(request_id, ())) for request_id in request_ids), default=0)
    payload: dict[str, Any] = {
        "path": path,
        "batch_size": len(request_ids),
        "request_ids": list(request_ids),
        "prompt_lengths": prompt_lengths,
        "decode_steps": decode_steps,
        "native_decode_steps": int(native_decode_steps),
        "serial_decode_fallback": len(request_ids) > 1 and not native_batch,
        "native_compact_prefill": False,
        "native_caware_decode": bool(native_batch),
        "native_sampler_rows": bool(native_batch),
        "throughput_claim_eligible": bool(native_batch and native_decode_steps > 0),
        "sampler_plan_metadata": [
            {
                "active_processors": list(plan.active_processors),
                "sampler_fast_path_blockers": list(plan.fast_path_blockers),
                "native_gpu_available": bool(native_batch),
                **(
                    {"sampler_fallback_reason": plan.fallback_reason}
                    if plan.fallback_reason is not None
                    else {}
                ),
                "sampler_mode": plan.mode.value,
            }
            for _request_id in request_ids
        ],
    }
    if execution_paths:
        payload["native_execution_paths"] = dict(execution_paths)
    if scheduling:
        payload["continuous_scheduler"] = dict(scheduling)
    payload["scheduler_token_chunks"] = _gguf_scheduler_token_chunks(
        request_ids,
        prompt_rows_by_request,
        generated_ids_by_request,
        token_logprobs_by_request,
        tokenizer=tokenizer,
        request=request,
        plan=plan,
        execution_path=path,
        native_batch=native_batch,
    )
    return payload


def _gguf_execution_path(plan: Any) -> str:
    if plan.mode is SamplingMode.GREEDY_FAST:
        return "gguf_serial_greedy_decode"
    return "gguf_serial_host_sampler_decode"


def _gguf_scheduler_token_chunks(
    request_ids: tuple[int, ...],
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids_by_request: dict[int, list[int]],
    token_logprobs_by_request: dict[int, list[TokenLogprob]],
    *,
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    plan: Any,
    execution_path: str,
    native_batch: bool = False,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for request_id in request_ids:
        generated_ids = generated_ids_by_request.get(request_id, [])
        token_logprobs = token_logprobs_by_request.get(request_id, [])
        prefix: list[int] = []
        for token_index, token_id in enumerate(generated_ids):
            prefix.append(int(token_id))
            final = token_index == len(generated_ids) - 1
            token_logprob = token_logprobs[token_index] if token_index < len(token_logprobs) else None
            token_text = (
                token_logprob.token_text
                if token_logprob is not None
                else tokenizer.decode([int(token_id)])
            )
            chunk = GenerationStreamChunk(
                text=token_text,
                token_logprobs=(
                    (token_logprob,)
                    if token_logprob is not None and (request.logprobs or int(request.top_logprobs) > 0)
                    else ()
                ),
                finish_details=(
                    _gguf_finish_details(prefix, tokenizer, request)
                    if final
                    else None
                ),
                telemetry=_gguf_telemetry(
                    prompt_rows_by_request.get(request_id, []),
                    prefix,
                    request,
                    row_index=request_id,
                    request_id=str(request_id),
                    phase="answer",
                    execution_path=execution_path,
                    native_compact_prefill=False,
                    native_caware_decode=native_batch,
                    serial_decode_fallback=len(request_ids) > 1 and not native_batch,
                    native_sampler_rows=native_batch,
                ),
            )
            chunks.append(_gguf_scheduler_token_chunk_payload(request_id, token_index, int(token_id), chunk))
    return chunks


def _gguf_scheduler_token_chunk_payload(
    request_id: int,
    token_index: int,
    token_id: int,
    chunk: GenerationStreamChunk,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": int(request_id),
        "token_index": int(token_index),
        "token_id": int(token_id),
        "finished": chunk.finish_details is not None,
        "chunk": {"text": chunk.text},
    }
    if chunk.token_logprobs:
        payload["chunk"]["token_logprobs"] = [
            {
                "token_id": token.token_id,
                "token_text": token.token_text,
                "logprob": token.logprob,
                "top_logprobs": [
                    {"token_id": top_id, "token_text": top_text, "logprob": top_logprob}
                    for top_id, top_text, top_logprob in token.top_logprobs
                ],
            }
            for token in chunk.token_logprobs
        ]
    if chunk.finish_details is not None:
        payload["chunk"]["finish_details"] = chunk.finish_details.to_json_dict()
    if chunk.telemetry is not None:
        payload["chunk"]["telemetry"] = chunk.telemetry.to_json_dict()
    return payload


def _gguf_queue_json_object_close_if_needed(
    state: RowSamplingState,
    tokenizer: Qwen35GGUFTokenizer,
    token_text: str,
    *,
    remaining_tokens: int,
) -> None:
    state.observe_text_for_json_object_close(
        token_text,
        remaining_tokens=remaining_tokens,
        encode_text=lambda text: tuple(int(token) for token in tokenizer.encode(str(text))),
    )


def _gguf_telemetry(
    prompt_ids: list[int] | tuple[int, ...],
    generated_ids: list[int] | tuple[int, ...],
    request: GenerationRequest,
    *,
    row_index: int,
    request_id: str | None = None,
    sampling_state: RowSamplingState | None = None,
    phase: str | None = None,
    forced_sample: Any | None = None,
    full_vocab_logits_d2h: bool | None = None,
    logits_d2h_bytes: int | None = None,
    execution_path: str | None = None,
    native_compact_prefill: bool | None = None,
    native_caware_decode: bool | None = None,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool | None = None,
) -> GenerationTelemetry:
    plan = _gguf_sampler_plan(request)
    state_payload = _gguf_decode_state_from_sampling_state(sampling_state)
    forced_token_id, forced_token_reason, forced_tokens_remaining = _gguf_forced_token_metadata(forced_sample)
    return GenerationTelemetry.from_decode_counts(
        request_id=request_id,
        row_index=row_index,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        phase=phase or state_payload.get("phase", "done"),
        reasoning_tokens=int(state_payload.get("reasoning_tokens", 0)),
        answer_tokens=int(state_payload.get("answer_tokens", 0)),
        forced_tokens_pending=tuple(state_payload.get("forced_tokens_pending", ())),
        forced_token_id=forced_token_id,
        forced_token_reason=forced_token_reason,
        forced_tokens_remaining=forced_tokens_remaining,
        post_thinking_forced_tokens_pending=tuple(state_payload.get("post_thinking_forced_tokens_pending", ())),
        post_thinking_forced_token_reason=state_payload.get("post_thinking_forced_token_reason"),
        force_sequence_completion_token_sequences=tuple(
            tuple(sequence) for sequence in state_payload.get("force_sequence_completion_token_sequences", ())
        ),
        force_sequence_completion_reason=state_payload.get("force_sequence_completion_reason"),
        budget_pressure=state_payload.get("budget_pressure"),
        sampler_mode=plan.mode.value,
        stop_suffix_state=_gguf_stop_suffix_state(generated_ids, request.stop_token_sequences),
        active_processors=plan.active_processors,
        sampler_fast_path_blockers=plan.fast_path_blockers,
        sampler_fallback_reason=plan.fallback_reason,
        full_vocab_logits_d2h=full_vocab_logits_d2h,
        logits_d2h_bytes=logits_d2h_bytes,
        execution_path=execution_path,
        native_compact_prefill=native_compact_prefill,
        native_caware_decode=native_caware_decode,
        serial_decode_fallback=serial_decode_fallback,
        native_sampler_rows=native_sampler_rows,
    )


def _gguf_forced_token_metadata(sample: Any | None) -> tuple[int | None, str | None, int | None]:
    if sample is None or not bool(getattr(sample, "forced", False)):
        return None, None, None
    return (
        int(getattr(sample, "token_id")),
        None if getattr(sample, "forced_reason", None) is None else str(getattr(sample, "forced_reason")),
        max(0, int(getattr(sample, "forced_tokens_remaining", 0))),
    )


def _gguf_decode_state_from_sampling_state(state: RowSamplingState | None) -> dict[str, Any]:
    if state is None:
        return {}
    payload: dict[str, Any] = {}
    if state.forced_tokens:
        payload["forced_tokens_pending"] = state.forced_tokens
    if state.post_thinking_forced_tokens_pending.pending_tokens:
        payload["post_thinking_forced_tokens_pending"] = state.post_thinking_forced_tokens_pending.pending_tokens
    if state.post_thinking_forced_token_reason is not None:
        payload["post_thinking_forced_token_reason"] = state.post_thinking_forced_token_reason
    if state.force_sequence_completion_token_sequences:
        payload["force_sequence_completion_token_sequences"] = state.force_sequence_completion_token_sequences
    if state.force_sequence_completion_reason is not None:
        payload["force_sequence_completion_reason"] = state.force_sequence_completion_reason
    budget = state.thinking_budget
    if budget is None:
        return payload
    payload["phase"] = str(budget.phase)
    payload["reasoning_tokens"] = int(budget.reasoning_tokens)
    payload["answer_tokens"] = int(budget.answer_tokens)
    forced_reason = getattr(budget.forced_tokens, "reason", None)
    pressure = "hard_close" if forced_reason == "thinking_hard_close" else budget.budget_pressure
    if pressure is not None:
        payload["budget_pressure"] = str(pressure)
    return payload


def _gguf_stop_suffix_state(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> dict[str, Any] | None:
    payload = token_sequence_state_for_tokens(generated_ids, stop_token_sequences).to_json_dict()
    return payload or None


def _gguf_finished(
    generated_ids: list[int] | tuple[int, ...],
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
) -> bool:
    if not generated_ids:
        return False
    token_id = int(generated_ids[-1])
    if not request.ignore_eos and int(token_id) == int(tokenizer.eos_token_id):
        return True
    if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
        return True
    for sequence in request.stop_token_sequences:
        if len(sequence) <= 0 or len(sequence) > len(generated_ids):
            continue
        if tuple(int(token) for token in generated_ids[-len(sequence) :]) == sequence:
            return True
    return False


def _gguf_finish_details(
    generated_ids: list[int] | tuple[int, ...],
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    state: RowSamplingState | None = None,
) -> FinishDetails:
    details: FinishDetails
    if generated_ids:
        token_id = int(generated_ids[-1])
        if not request.ignore_eos and int(token_id) == int(tokenizer.eos_token_id):
            details = FinishDetails(reason="eos", eos_token_id=token_id, sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
        if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
            details = FinishDetails(reason="stop", stop_sequence=(token_id,), sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
        sequence = _gguf_stop_sequence_match(generated_ids, request.stop_token_sequences)
        if sequence:
            details = FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
    if len(generated_ids) >= max(0, int(request.max_tokens)):
        details = FinishDetails(reason="length", length_limit=request.max_tokens, sampler_mode=_sampler_mode_value(request))
        return finish_details_with_sampling_state(details, state)
    details = FinishDetails(reason="stop", sampler_mode=_sampler_mode_value(request))
    return finish_details_with_sampling_state(details, state)


def _gguf_stop_sequence_match(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return token_sequence_state_for_tokens(generated_ids, stop_token_sequences).matched_sequence


def _sampler_mode_value(request: GenerationRequest) -> str:
    return _gguf_sampler_plan(request).mode.value


def _gguf_sampler_plan(request: GenerationRequest):
    return plan_sampler(request, native_gpu_requested=_native_gpu_sampler_requested())


def _native_gpu_sampler_requested() -> bool:
    value = os.environ.get("HIPENGINE_QWEN35_NATIVE_SAMPLER")
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


def _session_uses_host_routed_decode(session: Qwen35GGUFResidentSession) -> bool:
    """Return True for GGUF paths whose decode step cannot be graph-captured yet."""

    _ = session
    return False


def make_qwen35_gguf_bringup_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


def make_qwen35_gguf_ud_q3_k_m_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    """Select the exact fully-bulk prefill plugins for UD-Q3_K_M."""

    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        bulk_prefill_attention_mode="bulk",
        prefill_quant="gguf_ud_q3_k_m",
        prefill_attn_aotriton_min_tokens=0,
        native_batch_decode=True,
    )


for _model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
    for _quant in (
        "gguf_q4_k_m",
        "gguf_q8_0",
        "gguf_q4_1",
        "gguf_ud_q4_k_xl",
    ):
        register_text_generator(
            model=_model,
            backend="hip_gfx1100",
            quant=_quant,
            factory=make_qwen35_gguf_bringup_generator,
        )
    register_text_generator(
        model=_model,
        backend="hip_gfx1100",
        quant="gguf_ud_q3_k_m",
        factory=make_qwen35_gguf_ud_q3_k_m_generator,
    )


__all__ = [
    "Qwen35GGUFBringupGenerator",
    "make_qwen35_gguf_bringup_generator",
    "make_qwen35_gguf_ud_q3_k_m_generator",
]
