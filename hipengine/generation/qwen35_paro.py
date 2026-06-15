"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, ClassVar

from hipengine.generation.batch_scheduler import GeneratedToken, PerRowSamplingParams, ResidentBatchScheduler
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
    clone_thinking_budget_state,
    plan_sampler,
    row_seed_for_index,
    thinking_budget_state_from_params,
)
from hipengine.kvcache import resolve_kv_policy
from hipengine.loading import WeightIndex
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
    _decode_token_cached,
    _select_token,
)


@dataclass
class Qwen35ParoOneTokenGenerator:
    """Greedy Qwen3.5/PARO generator backed by resident c=1 execution.

    The implementation is still serial across prompts, but each prompt uses the
    resident single-request native prefill path followed by multi-token
    autoregressive decode using the resident HIP layer chain.
    """

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "auto"
    lm_head_chunk: int = 4096
    _runner: Qwen35ParoNextTokenRunner | None = field(default=None, init=False, repr=False)
    _session: Qwen35ParoResidentSession | None = field(default=None, init=False, repr=False)
    _session_capacity: int = field(default=0, init=False, repr=False)
    _session_batch_size: int = field(default=0, init=False, repr=False)
    _session_kv_key: tuple[str, str, str, int] | None = field(default=None, init=False, repr=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)
    supports_stream_logprobs: ClassVar[bool] = True

    def generate(self, request: GenerationRequest) -> list[str]:
        outputs = self.generate_detailed(request)
        return [output.text for output in outputs]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        native_gpu_available = _native_gpu_sampler_route_available(prompt_count=len(request.prompts))
        plan = plan_sampler(
            request,
            native_gpu_available=native_gpu_available,
            native_gpu_requested=_native_gpu_sampler_requested(),
        )
        if request.max_tokens == 0:
            self.last_batch_generation = None
            self.last_generation_outputs = tuple(
                GenerationOutput(
                    text="",
                    finish_details=_finish_details_for_tokens(
                        None,
                        (),
                        ignore_eos=request.ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=request.max_tokens,
                        sampler_mode=plan.mode.value,
                    ),
                )
                for _prompt in request.prompts
            )
            return list(self.last_generation_outputs)
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        if len(request.prompts) == 1:
            self.last_batch_generation = None
            if plan.mode is SamplingMode.GREEDY_FAST:
                output = self._generate_one(
                    runner,
                    request.prompts[0],
                    request.max_tokens,
                    ignore_eos=request.ignore_eos,
                    kv_policy=kv_policy,
                    sampler_mode=plan.mode.value,
                    deadline_at=request.deadline_at,
                    cancellation_token=request.cancellation_token,
                )
                self.last_generation_outputs = (output,)
                return list(self.last_generation_outputs)
            output = self._generate_one_sampled(
                runner,
                request.prompts[0],
                request.max_tokens,
                request=request,
                row_index=0,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
                plan=plan,
            )
            self.last_generation_outputs = (output,)
            return [output]
        if plan.mode is not SamplingMode.GREEDY_FAST:
            self.last_generation_outputs = tuple(
                self._generate_batch_sampled(
                    runner,
                    request.prompts,
                    request.max_tokens,
                    request=request,
                    ignore_eos=request.ignore_eos,
                    kv_policy=kv_policy,
                )
            )
            return list(self.last_generation_outputs)
        outputs = self._generate_batch(
            runner,
            request.prompts,
            request.max_tokens,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
            sampler_mode=plan.mode.value,
            deadline_at=request.deadline_at,
            cancellation_token=request.cancellation_token,
        )
        self.last_generation_outputs = tuple(outputs)
        return list(self.last_generation_outputs)

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int:
        params = sampling_params
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            getattr(params, "kv_storage", "auto"),
            scale_dtype=getattr(params, "kv_scale_dtype", "fp16"),
            scale_granularity=getattr(params, "kv_scale_granularity", "per_token_head"),
        )
        auto_context_length = max_sequence_length is None
        if auto_context_length:
            requested_length = int(getattr(runner.config, "max_position_embeddings", 0) or 0)
            if requested_length <= 0:
                requested_length = _session_capacity_for(1)
        else:
            if int(max_sequence_length) <= 0:
                raise ValueError("max_sequence_length must be positive")
            requested_length = int(max_sequence_length)
        session_capacity = _session_capacity_for(requested_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
            auto_context_length=auto_context_length,
        )
        return int(getattr(session, "max_sequence_length", self._session_capacity))

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        params = sampling_params
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            getattr(params, "kv_storage", "auto"),
            scale_dtype=getattr(params, "kv_scale_dtype", "fp16"),
            scale_granularity=getattr(params, "kv_scale_granularity", "per_token_head"),
        )
        required_sequence_length = max(1, int(max_prompt_tokens)) + max(0, int(max_new_tokens)) + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
            max_batch_size=max_batch_size,
        )
        return session.prepare_request_scratch(
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            max_batch_size=max_batch_size,
            release_after_probe=release_after_probe,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> tuple[int, ...]:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), str(text), None)
        return tuple(int(token) for token in prompt_ids)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield chunk.text

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        native_gpu_available = _native_gpu_sampler_route_available(prompt_count=1)
        plan = plan_sampler(
            request,
            native_gpu_available=native_gpu_available,
            native_gpu_requested=_native_gpu_sampler_requested(),
        )
        if request.max_tokens == 0:
            return
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        if plan.mode is SamplingMode.GREEDY_FAST:
            yield from self._stream_one(
                runner,
                request.prompts[0],
                request.max_tokens,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
                deadline_at=request.deadline_at,
                cancellation_token=request.cancellation_token,
            )
            return
        yield from self._stream_one_sampled(
            runner,
            request.prompts[0],
            request.max_tokens,
            request=request,
            row_index=0,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
            plan=plan,
        )

    def _generate_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        sampler_mode: str,
        deadline_at: float | None,
        cancellation_token: Any | None,
    ) -> GenerationOutput:
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        generated_text: list[str] = []
        generated_token_ids: list[int] = []
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        next_result = session.prefill_native(prompt_ids, sample=True)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        generated_text.append(next_result.token_text)
        generated_token_ids.append(int(next_result.token_id))
        if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
            return GenerationOutput(
                text="".join(generated_text),
                finish_details=_finish_details_for_tokens(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=(),
                    stop_token_sequences=(),
                    max_tokens=max_tokens,
                    sampler_mode=sampler_mode,
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=0,
                    sampler_mode=sampler_mode,
                    stop_token_sequences=(),
                ),
            )

        remaining = max_tokens - 1
        if remaining:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            with session.capture_decode_graph(
                position=len(prompt_ids),
                steps_per_replay=1,
                max_replay_steps=remaining,
                record_steps=remaining,
            ) as graph:
                raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                graph.replay(remaining)
                raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                token_ids = graph.read_generated_token_ids(remaining)
                raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            for token_id in token_ids:
                generated_text.append(_decode_token_cached(session.tokenizer, token_id))
                generated_token_ids.append(int(token_id))
                if not ignore_eos and _is_eos(session.tokenizer, token_id):
                    break
        return GenerationOutput(
            text="".join(generated_text),
            finish_details=_finish_details_for_tokens(
                session.tokenizer,
                generated_token_ids,
                ignore_eos=ignore_eos,
                stop_token_ids=(),
                stop_token_sequences=(),
                max_tokens=max_tokens,
                sampler_mode=sampler_mode,
            ),
            telemetry=_telemetry_for_tokens(
                prompt_ids,
                generated_token_ids,
                row_index=0,
                sampler_mode=sampler_mode,
                stop_token_sequences=(),
            ),
        )

    def _generate_one_sampled(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        request: GenerationRequest,
        row_index: int,
        ignore_eos: bool,
        kv_policy,
        plan,
    ) -> GenerationOutput:
        raise_if_generation_deadline_expired(request)
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        sampling_request = _request_with_tokenizer_eos(request, session.tokenizer)
        state = _row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        _configure_sampled_session(session, sampling_request, state, plan=plan)
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            plan,
            vocab_size=getattr(session, "vocab_size", None),
        )
        generated_text: list[str] = []
        generated_token_ids: list[int] = []
        generated_steps: list[Qwen35ParoAutoregressiveStepResult] = []
        try:
            raise_if_generation_deadline_expired(request)
            next_result = session.prefill_native(prompt_ids, sample=True)
            raise_if_generation_deadline_expired(request)
            if next_result is None:
                raise RuntimeError("native prefill did not produce next-token logits")
            generated_text.append(next_result.token_text)
            generated_token_ids.append(int(next_result.token_id))
            generated_steps.append(next_result)
            _queue_json_object_close_if_needed(
                state,
                session.tokenizer,
                next_result.token_text,
                remaining_tokens=max_tokens - len(generated_token_ids),
            )
            if _is_finished(
                session.tokenizer,
                generated_token_ids,
                ignore_eos=ignore_eos,
                stop_token_ids=request.stop_token_ids,
                stop_token_sequences=request.stop_token_sequences,
            ):
                return _generation_output_from_steps(
                    session.tokenizer,
                    generated_steps,
                    finish_details=_finish_details_for_tokens(
                        session.tokenizer,
                        generated_token_ids,
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=max_tokens,
                        sampler_mode=plan.mode.value,
                        sampling_state=state,
                    ),
                    telemetry=_telemetry_for_tokens(
                        prompt_ids,
                        generated_token_ids,
                        row_index=row_index,
                        sampler_mode=plan.mode.value,
                        stop_token_sequences=request.stop_token_sequences,
                        active_processors=plan.active_processors,
                        sampler_fast_path_blockers=plan.fast_path_blockers,
                        sampler_fallback_reason=plan.fallback_reason,
                        sampling_state=state,
                        forced_sample=next_result,
                        full_vocab_logits_d2h=full_vocab_logits_d2h,
                        logits_d2h_bytes=logits_d2h_bytes,
                    ),
                )

            current_token_id = int(next_result.token_id)
            for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
                raise_if_generation_deadline_expired(request)
                result = session.step(current_token_id, position=position, sample=True)
                raise_if_generation_deadline_expired(request)
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                generated_text.append(result.token_text)
                generated_token_ids.append(int(result.token_id))
                generated_steps.append(result)
                _queue_json_object_close_if_needed(
                    state,
                    session.tokenizer,
                    result.token_text,
                    remaining_tokens=max_tokens - len(generated_token_ids),
                )
                current_token_id = int(result.token_id)
                if _is_finished(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=request.stop_token_ids,
                    stop_token_sequences=request.stop_token_sequences,
                ):
                    break
            return _generation_output_from_steps(
                session.tokenizer,
                generated_steps,
                finish_details=_finish_details_for_tokens(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=request.stop_token_ids,
                    stop_token_sequences=request.stop_token_sequences,
                    max_tokens=max_tokens,
                    sampler_mode=plan.mode.value,
                    sampling_state=state,
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=row_index,
                    sampler_mode=plan.mode.value,
                    stop_token_sequences=request.stop_token_sequences,
                    active_processors=plan.active_processors,
                    sampler_fast_path_blockers=plan.fast_path_blockers,
                    sampler_fallback_reason=plan.fallback_reason,
                    sampling_state=state,
                    forced_sample=generated_steps[-1] if generated_steps else None,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                ),
            )
        finally:
            _configure_sampled_session(session, None, None, plan=plan)

    def _generate_batch(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[str, ...],
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        sampler_mode: str,
        deadline_at: float | None,
        cancellation_token: Any | None,
    ) -> list[GenerationOutput]:
        """Generate a prompt list through the scheduler-owned c>N path.

        Native compact prefill runs all admitted rows together when their block
        table shapes permit it. Decode remains the explicit serial slot bridge
        until native c-aware replay lands; keep ``last_batch_generation`` clear
        about that so prompt-list batching is not mistaken for a retained c>N
        throughput path.
        """

        prompt_rows: list[list[int]] = []
        for prompt in prompts:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append([int(token) for token in prompt_ids])
        batch_size = len(prompt_rows)
        required_sequence_length = max(len(row) for row in prompt_rows) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            max_batch_size=batch_size,
            kv_policy=kv_policy,
        )
        scheduler = ResidentBatchScheduler(capacity=batch_size)
        request_ids = tuple(
            scheduler.submit(row, max_new_tokens=max(0, max_tokens - 1))
            for row in prompt_rows
        )
        prompt_rows_by_request = dict(zip(request_ids, prompt_rows, strict=True))
        admitted = scheduler.admit_pending()
        if admitted != request_ids:
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")

        output_parts: dict[int, list[str]] = {request_id: [] for request_id in request_ids}
        generated_ids: dict[int, list[int]] = {request_id: [] for request_id in request_ids}
        next_token_by_request: dict[int, int] = {}
        packed_slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(len(row) for row in prompt_rows),
            block_size=getattr(session, "block_size", 256),
        )
        prefill_slab_shapes: list[dict[str, Any]] = []
        for slab in packed_slabs:
            prefill_slab_shapes.append(
                {
                    "request_ids": list(slab.request_ids),
                    "slot_ids": list(slab.physical_slot_ids),
                    "rows": slab.rows,
                    "request_count": slab.request_count,
                    "block_count": slab.block_count,
                }
            )
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            results = session.prefill_native_packed(slab, sample=True)
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            if len(results) != slab.request_count:
                raise RuntimeError(
                    "packed prefill returned "
                    f"{len(results)} results for {slab.request_count} requests"
                )
            for request_id, result in zip(slab.request_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("packed native prefill did not produce next-token logits")
                output_parts[request_id].append(result.token_text)
                generated_ids[request_id].append(int(result.token_id))
                seed_finished = (
                    not ignore_eos and _is_eos(session.tokenizer, result.token_id)
                ) or max_tokens <= 1
                if seed_finished:
                    scheduler.record_generated(
                        (GeneratedToken(request_id, result.token_id, finished=True),)
                    )
                else:
                    next_token_by_request[request_id] = int(result.token_id)

        decode_steps = 0
        native_decode_steps = 0
        serial_decode_fallback = False
        while next_token_by_request:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            work = scheduler.next_decode_work()
            if work is None:
                raise RuntimeError("scheduler did not emit decode work")
            request_ids_for_step = tuple(
                request_id for request_id in work.request_ids if request_id in next_token_by_request
            )
            if not request_ids_for_step:
                raise RuntimeError("scheduler decode work did not include runnable requests")
            token_ids_for_step = [next_token_by_request[request_id] for request_id in request_ids_for_step]
            positions_for_step = [
                scheduler.active_batch.requests[request_id].context_len
                for request_id in request_ids_for_step
            ]
            slots_for_step = [
                scheduler.active_batch.slot_for(request_id)
                for request_id in request_ids_for_step
            ]
            compact_slots = tuple(slots_for_step) == tuple(range(len(slots_for_step)))
            use_native_decode = compact_slots and len(slots_for_step) > 1 and hasattr(session, "step_batch_native")
            if use_native_decode:
                try:
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    results = session.step_batch_native(
                        token_ids_for_step,
                        positions=positions_for_step,
                        slots=slots_for_step,
                        sample=True,
                    )
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    native_decode_steps += 1
                except NotImplementedError:
                    serial_decode_fallback = True
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    results = session.step_batch_serial(
                        token_ids_for_step,
                        positions=positions_for_step,
                        slots=slots_for_step,
                        sample=True,
                    )
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            else:
                serial_decode_fallback = serial_decode_fallback or len(slots_for_step) > 1
                raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                results = session.step_batch_serial(
                    token_ids_for_step,
                    positions=positions_for_step,
                    slots=slots_for_step,
                    sample=True,
                )
                raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            generated: list[GeneratedToken] = []
            for request_id, result in zip(request_ids_for_step, results, strict=True):
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                output_parts[request_id].append(result.token_text)
                generated_ids[request_id].append(int(result.token_id))
                next_token_by_request[request_id] = int(result.token_id)
                finished = not ignore_eos and _is_eos(session.tokenizer, result.token_id)
                generated.append(GeneratedToken(request_id, result.token_id, finished=finished))
            completed = scheduler.record_generated(generated)
            for done in completed:
                next_token_by_request.pop(done.request_id, None)
            decode_steps += 1

        native_decode_complete = decode_steps > 0 and native_decode_steps == decode_steps and not serial_decode_fallback
        batch_execution = session.batch_execution_metadata(
            scheduler_owned=True,
            native_decode=native_decode_complete,
        )
        self.last_batch_generation = {
            "path": "scheduler_native_packed_prefill_native_decode"
            if native_decode_complete
            else "scheduler_native_packed_prefill_serial_decode",
            "batch_size": batch_size,
            "request_ids": list(request_ids),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "packed_prefill_slabs": prefill_slab_shapes,
            "decode_steps": decode_steps,
            "native_decode_steps": native_decode_steps,
            "serial_decode_fallback": serial_decode_fallback,
            "native_compact_prefill": bool(
                getattr(batch_execution, "native_compact_prefill", False)
            ),
            "native_caware_decode": bool(getattr(batch_execution, "native_caware_decode", False)),
            "throughput_claim_eligible": bool(
                getattr(batch_execution, "throughput_claim_eligible", False)
            ),
        }
        self.last_batch_generation["scheduler_token_chunks"] = _batch_scheduler_token_chunks(
            request_ids,
            prompt_rows_by_request,
            generated_ids,
            output_parts,
            tokenizer=session.tokenizer,
            ignore_eos=ignore_eos,
            stop_token_ids=(),
            stop_token_sequences=(),
            max_tokens=max_tokens,
            sampler_mode=sampler_mode,
            execution_path=self.last_batch_generation["path"],
            native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
            native_caware_decode=self.last_batch_generation["native_caware_decode"],
            serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
        )
        return [
            GenerationOutput(
                text="".join(output_parts[request_id]),
                finish_details=_finish_details_for_tokens(
                    session.tokenizer,
                    generated_ids[request_id],
                    ignore_eos=ignore_eos,
                    stop_token_ids=(),
                    stop_token_sequences=(),
                    max_tokens=max_tokens,
                    sampler_mode=sampler_mode,
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_rows_by_request[request_id],
                    generated_ids[request_id],
                    row_index=request_id,
                    request_id=str(request_id),
                    sampler_mode=sampler_mode,
                    stop_token_sequences=(),
                    execution_path=self.last_batch_generation["path"],
                    native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
                    native_caware_decode=self.last_batch_generation["native_caware_decode"],
                    serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
                ),
            )
            for request_id in request_ids
        ]

    def _generate_batch_sampled(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[str, ...],
        max_tokens: int,
        *,
        request: GenerationRequest,
        ignore_eos: bool,
        kv_policy,
    ) -> list[GenerationOutput]:
        """Generate a sampled prompt list through scheduler-owned c>N state.

        Native packed prefill handles the prompt rows together, while decode uses
        the explicit serial slot bridge with per-slot host sampler state clones.
        The scheduler remains the owner of persistent row history.
        """

        prompt_rows: list[list[int]] = []
        for prompt in prompts:
            raise_if_generation_deadline_expired(request)
            _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
            raise_if_generation_deadline_expired(request)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append([int(token) for token in prompt_ids])
        batch_size = len(prompt_rows)
        required_sequence_length = max(len(row) for row in prompt_rows) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            max_batch_size=batch_size,
            kv_policy=kv_policy,
        )
        sampling_request = _request_with_tokenizer_eos(request, session.tokenizer)
        scheduler = ResidentBatchScheduler(capacity=batch_size)
        sampling = _per_row_sampling_params(sampling_request)
        request_ids = tuple(
            scheduler.submit(
                row,
                max_new_tokens=max(0, max_tokens - 1),
                sampling=sampling,
                sampling_row_index=index,
            )
            for index, row in enumerate(prompt_rows)
        )
        prompt_rows_by_request = dict(zip(request_ids, prompt_rows, strict=True))
        admitted = scheduler.admit_pending()
        if admitted != request_ids:
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
        native_sampler_requested = _native_gpu_sampler_requested()
        configure_native_rows = getattr(session, "configure_native_sampler_rows", None)
        native_sampler_rows_available = native_sampler_requested and callable(configure_native_rows)
        sampler_block = scheduler.sampler_params_block(request_ids)
        sampler_plans = dict(
            zip(
                request_ids,
                sampler_block.sampler_plans(
                    native_gpu_available=native_sampler_rows_available,
                    native_gpu_requested=native_sampler_requested,
                ),
                strict=True,
            )
        )
        sampler_plan_metadata = sampler_block.sampler_plan_metadata(
            native_gpu_available=native_sampler_rows_available,
            native_gpu_requested=native_sampler_requested
        )
        use_native_sampler_rows = native_sampler_rows_available and all(
            plan.mode is SamplingMode.GPU_SAMPLE for plan in sampler_plans.values()
        )

        output_steps: dict[int, list[Qwen35ParoAutoregressiveStepResult]] = {request_id: [] for request_id in request_ids}
        generated_ids: dict[int, list[int]] = {request_id: [] for request_id in request_ids}
        sampling_state_snapshots: dict[int, RowSamplingState] = {
            request_id: _clone_row_sampling_state(scheduler.sampler_state(request_id))
            for request_id in request_ids
        }
        sampling_state_step_snapshots: dict[int, list[RowSamplingState]] = {
            request_id: [] for request_id in request_ids
        }
        next_token_by_request: dict[int, int] = {}
        packed_slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(len(row) for row in prompt_rows),
            block_size=getattr(session, "block_size", 256),
        )
        prefill_slab_shapes: list[dict[str, Any]] = []
        if use_native_sampler_rows:
            configure_rows = configure_native_rows
            sampled_path = "scheduler_native_packed_prefill_serial_native_sampler_decode"
        else:
            configure_rows = getattr(session, "configure_host_sampler_rows", None)
            sampled_path = "scheduler_native_packed_prefill_serial_host_sampler_decode"
        if not callable(configure_rows):
            raise NotImplementedError("c>N sampled PARO batches require per-slot host sampler state")
        try:
            for slab in packed_slabs:
                prefill_slab_shapes.append(
                    {
                        "request_ids": list(slab.request_ids),
                        "slot_ids": list(slab.physical_slot_ids),
                        "rows": slab.rows,
                        "request_count": slab.request_count,
                        "block_count": slab.block_count,
                    }
                )
                configure_rows(sampling_request, _slot_sampler_state_clones(scheduler, slab.request_ids, slab.physical_slot_ids))
                raise_if_generation_deadline_expired(request)
                results = session.prefill_native_packed(slab, sample=True)
                raise_if_generation_deadline_expired(request)
                if len(results) != slab.request_count:
                    raise RuntimeError(
                        "packed prefill returned "
                        f"{len(results)} results for {slab.request_count} requests"
                    )
                generated: list[GeneratedToken] = []
                for request_id, result in zip(slab.request_ids, results, strict=True):
                    if result is None:
                        raise RuntimeError("packed native prefill did not produce next-token logits")
                    output_steps[request_id].append(result)
                    generated_ids[request_id].append(int(result.token_id))
                    snapshot = _clone_row_sampling_state(scheduler.sampler_state(request_id))
                    snapshot.observe(result.token_id)
                    sampling_state_snapshots[request_id] = snapshot
                    finished = max_tokens <= 1 or _is_finished(
                        session.tokenizer,
                        generated_ids[request_id],
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                    )
                    if finished:
                        sampling_state_step_snapshots[request_id].append(snapshot)
                        generated.append(GeneratedToken(request_id, result.token_id, finished=True))
                    else:
                        owner_state = scheduler.sampler_state(request_id)
                        owner_state.observe(result.token_id)
                        _queue_json_object_close_if_needed(
                            owner_state,
                            session.tokenizer,
                            result.token_text,
                            remaining_tokens=max_tokens - len(generated_ids[request_id]),
                        )
                        sampling_state_snapshots[request_id] = _clone_row_sampling_state(owner_state)
                        sampling_state_step_snapshots[request_id].append(sampling_state_snapshots[request_id])
                        next_token_by_request[request_id] = int(result.token_id)
                if generated:
                    completed_ids = {done.request_id for done in scheduler.record_generated(generated)}
                    for done in completed_ids:
                        next_token_by_request.pop(done, None)

            decode_steps = 0
            serial_decode_fallback = False
            while next_token_by_request:
                raise_if_generation_deadline_expired(request)
                work = scheduler.next_decode_work()
                if work is None:
                    raise RuntimeError("scheduler did not emit decode work")
                request_ids_for_step = tuple(
                    request_id for request_id in work.request_ids if request_id in next_token_by_request
                )
                if not request_ids_for_step:
                    raise RuntimeError("scheduler decode work did not include runnable requests")
                token_ids_for_step = [next_token_by_request[request_id] for request_id in request_ids_for_step]
                positions_for_step = [
                    scheduler.active_batch.requests[request_id].context_len
                    for request_id in request_ids_for_step
                ]
                slots_for_step = [scheduler.active_batch.slot_for(request_id) for request_id in request_ids_for_step]
                configure_rows(
                    sampling_request,
                    _slot_sampler_state_clones(scheduler, request_ids_for_step, slots_for_step),
                )
                raise_if_generation_deadline_expired(request)
                results = session.step_batch_serial(
                    token_ids_for_step,
                    positions=positions_for_step,
                    slots=slots_for_step,
                    sample=True,
                )
                raise_if_generation_deadline_expired(request)
                serial_decode_fallback = serial_decode_fallback or len(slots_for_step) > 1
                generated = []
                decode_results_by_request: dict[int, Qwen35ParoAutoregressiveStepResult] = {}
                for request_id, result in zip(request_ids_for_step, results, strict=True):
                    if result is None:
                        raise RuntimeError("decode step did not produce next-token logits")
                    output_steps[request_id].append(result)
                    generated_ids[request_id].append(int(result.token_id))
                    snapshot = _clone_row_sampling_state(scheduler.sampler_state(request_id))
                    snapshot.observe(result.token_id)
                    sampling_state_snapshots[request_id] = snapshot
                    finished = _is_finished(
                        session.tokenizer,
                        generated_ids[request_id],
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                    )
                    generated.append(GeneratedToken(request_id, result.token_id, finished=finished))
                    decode_results_by_request[int(request_id)] = result
                completed = scheduler.record_generated(generated)
                completed_ids = {done.request_id for done in completed}
                for done in completed_ids:
                    next_token_by_request.pop(done, None)
                for request_id, result in decode_results_by_request.items():
                    if request_id in completed_ids:
                        sampling_state_step_snapshots[request_id].append(sampling_state_snapshots[request_id])
                        continue
                    owner_state = scheduler.sampler_state(request_id)
                    _queue_json_object_close_if_needed(
                        owner_state,
                        session.tokenizer,
                        result.token_text,
                        remaining_tokens=max_tokens - len(generated_ids[request_id]),
                    )
                    sampling_state_snapshots[request_id] = _clone_row_sampling_state(owner_state)
                    sampling_state_step_snapshots[request_id].append(sampling_state_snapshots[request_id])
                    next_token_by_request[request_id] = int(result.token_id)
                decode_steps += 1
        finally:
            configure_rows(None, None)

        batch_execution = session.batch_execution_metadata(scheduler_owned=True, native_decode=False)
        self.last_batch_generation = {
            "path": sampled_path,
            "batch_size": batch_size,
            "request_ids": list(request_ids),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "packed_prefill_slabs": prefill_slab_shapes,
            "decode_steps": decode_steps,
            "native_decode_steps": 0,
            "serial_decode_fallback": serial_decode_fallback,
            "native_compact_prefill": bool(getattr(batch_execution, "native_compact_prefill", False)),
            "native_caware_decode": False,
            "native_sampler_rows": use_native_sampler_rows,
            "throughput_claim_eligible": False,
            "sampler_plan_metadata": [dict(row) for row in sampler_plan_metadata],
        }
        self.last_batch_generation["scheduler_token_chunks"] = _sampled_batch_scheduler_token_chunks(
            request_ids,
            prompt_rows_by_request,
            output_steps,
            sampling_state_step_snapshots,
            tokenizer=session.tokenizer,
            vocab_size=getattr(session, "vocab_size", None),
            request=sampling_request,
            plans=sampler_plans,
            execution_path=self.last_batch_generation["path"],
            native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
            native_caware_decode=self.last_batch_generation["native_caware_decode"],
            serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
            native_sampler_rows=self.last_batch_generation["native_sampler_rows"],
        )
        outputs: list[GenerationOutput] = []
        for request_id in request_ids:
            plan = sampler_plans[request_id]
            sampler_mode = plan.mode.value
            full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
                plan,
                vocab_size=getattr(session, "vocab_size", None),
            )
            outputs.append(
                _generation_output_from_steps(
                    session.tokenizer,
                    output_steps[request_id],
                    finish_details=_finish_details_for_tokens(
                        session.tokenizer,
                        generated_ids[request_id],
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=request.max_tokens,
                        sampler_mode=sampler_mode,
                        sampling_state=sampling_state_snapshots.get(request_id),
                    ),
                    telemetry=_telemetry_for_tokens(
                        prompt_rows_by_request[request_id],
                        generated_ids[request_id],
                        row_index=request_id,
                        request_id=str(request_id),
                        sampler_mode=sampler_mode,
                        stop_token_sequences=request.stop_token_sequences,
                        active_processors=plan.active_processors,
                        sampler_fast_path_blockers=plan.fast_path_blockers,
                        sampler_fallback_reason=plan.fallback_reason,
                        sampling_state=sampling_state_snapshots.get(request_id),
                        forced_sample=output_steps[request_id][-1] if output_steps[request_id] else None,
                        full_vocab_logits_d2h=full_vocab_logits_d2h,
                        logits_d2h_bytes=logits_d2h_bytes,
                        execution_path=self.last_batch_generation["path"],
                        native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
                        native_caware_decode=self.last_batch_generation["native_caware_decode"],
                        serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
                        native_sampler_rows=self.last_batch_generation["native_sampler_rows"],
                    ),
                )
            )
        return outputs

    def _stream_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        deadline_at: float | None,
        cancellation_token: Any | None,
    ) -> Iterator[GenerationStreamChunk]:
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        next_result = session.prefill_native(prompt_ids, sample=True)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        generated_token_ids = [int(next_result.token_id)]
        finished = not ignore_eos and _is_eos(session.tokenizer, next_result.token_id)
        yield GenerationStreamChunk(
            next_result.token_text,
            finish_details=(
                _finish_details_for_tokens(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=(),
                    stop_token_sequences=(),
                    max_tokens=max_tokens,
                    sampler_mode=SamplingMode.GREEDY_FAST.value,
                )
                if finished or len(generated_token_ids) >= max_tokens
                else None
            ),
            telemetry=_telemetry_for_tokens(
                prompt_ids,
                generated_token_ids,
                row_index=0,
                sampler_mode=SamplingMode.GREEDY_FAST.value,
                phase="answer",
                stop_token_sequences=(),
            ),
        )
        if finished:
            return

        current_token_id = next_result.token_id
        for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            result = session.step(current_token_id, position=position, sample=True)
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            if result is None:
                raise RuntimeError("decode step did not produce next-token logits")
            generated_token_ids.append(int(result.token_id))
            finished = not ignore_eos and _is_eos(session.tokenizer, result.token_id)
            yield GenerationStreamChunk(
                result.token_text,
                finish_details=(
                    _finish_details_for_tokens(
                        session.tokenizer,
                        generated_token_ids,
                        ignore_eos=ignore_eos,
                        stop_token_ids=(),
                        stop_token_sequences=(),
                        max_tokens=max_tokens,
                        sampler_mode=SamplingMode.GREEDY_FAST.value,
                    )
                    if finished or len(generated_token_ids) >= max_tokens
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=0,
                    sampler_mode=SamplingMode.GREEDY_FAST.value,
                    phase="answer",
                    stop_token_sequences=(),
                ),
            )
            current_token_id = result.token_id
            if finished:
                return

    def _stream_one_sampled(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        request: GenerationRequest,
        row_index: int,
        ignore_eos: bool,
        kv_policy,
        plan,
    ) -> Iterator[GenerationStreamChunk]:
        raise_if_generation_deadline_expired(request)
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        sampling_request = _request_with_tokenizer_eos(request, session.tokenizer)
        state = _row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        _configure_sampled_session(session, sampling_request, state, plan=plan)
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            plan,
            vocab_size=getattr(session, "vocab_size", None),
        )
        generated_token_ids: list[int] = []
        live_phase = None if state.thinking_budget is not None else "answer"
        try:
            raise_if_generation_deadline_expired(request)
            next_result = session.prefill_native(prompt_ids, sample=True)
            raise_if_generation_deadline_expired(request)
            if next_result is None:
                raise RuntimeError("native prefill did not produce next-token logits")
            generated_token_ids.append(int(next_result.token_id))
            _queue_json_object_close_if_needed(
                state,
                session.tokenizer,
                next_result.token_text,
                remaining_tokens=max_tokens - len(generated_token_ids),
            )
            finished = _is_finished(
                session.tokenizer,
                generated_token_ids,
                ignore_eos=ignore_eos,
                stop_token_ids=sampling_request.stop_token_ids,
                stop_token_sequences=sampling_request.stop_token_sequences,
            )
            yield GenerationStreamChunk(
                next_result.token_text,
                token_logprobs=_stream_token_logprobs_from_step(session.tokenizer, next_result, sampling_request),
                finish_details=(
                    _finish_details_for_tokens(
                        session.tokenizer,
                        generated_token_ids,
                        ignore_eos=ignore_eos,
                        stop_token_ids=sampling_request.stop_token_ids,
                        stop_token_sequences=sampling_request.stop_token_sequences,
                        max_tokens=max_tokens,
                        sampler_mode=plan.mode.value,
                        sampling_state=state,
                    )
                    if finished or len(generated_token_ids) >= max_tokens
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=row_index,
                    sampler_mode=plan.mode.value,
                    stop_token_sequences=sampling_request.stop_token_sequences,
                    phase=live_phase,
                    active_processors=plan.active_processors,
                    sampler_fast_path_blockers=plan.fast_path_blockers,
                    sampler_fallback_reason=plan.fallback_reason,
                    sampling_state=state,
                    forced_sample=next_result,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                ),
            )
            if finished:
                return

            current_token_id = int(next_result.token_id)
            for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
                raise_if_generation_deadline_expired(request)
                result = session.step(current_token_id, position=position, sample=True)
                raise_if_generation_deadline_expired(request)
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                generated_token_ids.append(int(result.token_id))
                _queue_json_object_close_if_needed(
                    state,
                    session.tokenizer,
                    result.token_text,
                    remaining_tokens=max_tokens - len(generated_token_ids),
                )
                finished = _is_finished(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=sampling_request.stop_token_ids,
                    stop_token_sequences=sampling_request.stop_token_sequences,
                )
                yield GenerationStreamChunk(
                    result.token_text,
                    token_logprobs=_stream_token_logprobs_from_step(session.tokenizer, result, sampling_request),
                    finish_details=(
                        _finish_details_for_tokens(
                            session.tokenizer,
                            generated_token_ids,
                            ignore_eos=ignore_eos,
                            stop_token_ids=sampling_request.stop_token_ids,
                            stop_token_sequences=sampling_request.stop_token_sequences,
                            max_tokens=max_tokens,
                            sampler_mode=plan.mode.value,
                            sampling_state=state,
                        )
                        if finished or len(generated_token_ids) >= max_tokens
                        else None
                    ),
                    telemetry=_telemetry_for_tokens(
                        prompt_ids,
                        generated_token_ids,
                        row_index=row_index,
                        sampler_mode=plan.mode.value,
                        stop_token_sequences=sampling_request.stop_token_sequences,
                        phase=live_phase,
                        active_processors=plan.active_processors,
                        sampler_fast_path_blockers=plan.fast_path_blockers,
                        sampler_fallback_reason=plan.fallback_reason,
                        sampling_state=state,
                        forced_sample=result,
                        full_vocab_logits_d2h=full_vocab_logits_d2h,
                        logits_d2h_bytes=logits_d2h_bytes,
                    ),
                )
                current_token_id = int(result.token_id)
                if finished:
                    return
        finally:
            _configure_sampled_session(session, None, None, plan=plan)

    def _get_runner(self) -> Qwen35ParoNextTokenRunner:
        if self._runner is None:
            self._runner = Qwen35ParoNextTokenRunner(
                self.model_path,
                index=self.weight_index,
                backend=self.backend,
            )
        return self._runner

    def _get_session(
        self,
        runner: Qwen35ParoNextTokenRunner,
        *,
        max_sequence_length: int,
        kv_policy,
        auto_context_length: bool = False,
        max_batch_size: int = 1,
    ) -> Qwen35ParoResidentSession:
        kv_key = (
            kv_policy.storage_dtype.value,
            kv_policy.scale_dtype.value,
            kv_policy.scale_granularity,
            int(kv_policy.block_size),
        )
        batch_size = max(1, int(max_batch_size))
        capacity_ok = self._session_capacity >= max_sequence_length or bool(auto_context_length)
        batch_ok = self._session_batch_size >= batch_size
        if (
            self._session is None
            or not capacity_ok
            or not batch_ok
            or self._session_kv_key != kv_key
        ):
            self.close()
            session_kwargs = {
                "max_sequence_length": max_sequence_length,
                "kv_policy": kv_policy.create_policy(),
                "kv_scale_dtype": kv_policy.scale_dtype,
                "kv_scale_granularity": kv_policy.scale_granularity,
            }
            if auto_context_length:
                session_kwargs["auto_context_length"] = True
            if batch_size > 1:
                session_kwargs["max_batch_size"] = batch_size
            self._session = Qwen35ParoResidentSession(runner, **session_kwargs)
            self._session_capacity = int(
                getattr(self._session, "max_sequence_length", max_sequence_length)
            )
            self._session_batch_size = int(getattr(self._session, "max_batch_size", batch_size))
            self._session_kv_key = kv_key
        else:
            self._session.reset()
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._session_capacity = 0
        self._session_batch_size = 0
        self._session_kv_key = None


def _per_row_sampling_params(request: GenerationRequest) -> PerRowSamplingParams:
    return PerRowSamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        logit_bias=request.logit_bias,
        suppress_tokens=request.suppress_token_ids,
        min_tokens=request.min_tokens,
        eos_token_id=request.eos_token_id,
        ignore_eos=request.ignore_eos,
        seed=request.seed,
        stop_tokens=request.stop_token_ids,
        stop_token_sequences=request.stop_token_sequences,
        forced_tokens_pending=request.forced_tokens_pending,
        forced_token_reason=request.forced_token_reason,
        post_thinking_forced_tokens_pending=request.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=request.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=request.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=request.force_sequence_completion_reason,
        json_object_close_forcing=request.json_object_close_forcing,
        thinking_close_token_ids=request.thinking_close_token_ids,
        thinking_hard_token_cap=request.thinking_hard_token_cap,
        thinking_soft_close_window=request.thinking_soft_close_window,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs,
    )


def _slot_sampler_state_clones(
    scheduler: ResidentBatchScheduler,
    request_ids: tuple[int, ...] | list[int],
    slots: tuple[int, ...] | list[int],
) -> dict[int, RowSamplingState]:
    return {
        int(slot): _clone_row_sampling_state(scheduler.sampler_state(int(request_id)))
        for request_id, slot in zip(request_ids, slots, strict=True)
    }


def _clone_row_sampling_state(state: RowSamplingState) -> RowSamplingState:
    thinking_budget = clone_thinking_budget_state(state.thinking_budget)
    if thinking_budget is not None:
        return RowSamplingState(
            prompt_tokens=state.prompt_tokens,
            seed=state.seed,
            request_id=state.request_id,
            row_index=state.row_index,
            generated_tokens=tuple(state.generated_tokens),
            step_index=state.step_index,
            stop_token_sequences=state.stop_token_sequences,
            post_thinking_forced_tokens_pending=state.post_thinking_forced_tokens_pending.pending_tokens,
            post_thinking_forced_token_reason=state.post_thinking_forced_token_reason,
            force_sequence_completion_token_sequences=state.force_sequence_completion_token_sequences,
            force_sequence_completion_reason=state.force_sequence_completion_reason,
            json_object_close_forcing=state.json_object_close_forcing,
            thinking_budget=thinking_budget,
        )
    return RowSamplingState(
        prompt_tokens=state.prompt_tokens,
        seed=state.seed,
        request_id=state.request_id,
        row_index=state.row_index,
        generated_tokens=tuple(state.generated_tokens),
        step_index=state.step_index,
        stop_token_sequences=state.stop_token_sequences,
        forced_tokens_pending=state.forced_tokens,
        forced_token_reason=state.forced_token_reason,
        post_thinking_forced_tokens_pending=state.post_thinking_forced_tokens_pending.pending_tokens,
        post_thinking_forced_token_reason=state.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=state.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=state.force_sequence_completion_reason,
        json_object_close_forcing=state.json_object_close_forcing,
    )


def _generation_output_from_steps(
    tokenizer: Any,
    steps: list[Qwen35ParoAutoregressiveStepResult] | tuple[Qwen35ParoAutoregressiveStepResult, ...],
    *,
    finish_details: FinishDetails,
    telemetry: GenerationTelemetry | None = None,
) -> GenerationOutput:
    tokens = tuple(_token_logprob_from_step(tokenizer, step) for step in steps)
    return GenerationOutput(
        text="".join(step.token_text for step in steps),
        token_logprobs=tokens,
        finish_details=finish_details,
        telemetry=telemetry,
    )


def _batch_scheduler_token_chunks(
    request_ids: tuple[int, ...],
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids: dict[int, list[int]],
    generated_texts: dict[int, list[str]],
    *,
    tokenizer: Any,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
    max_tokens: int,
    sampler_mode: str,
    execution_path: str | None,
    native_compact_prefill: bool | None,
    native_caware_decode: bool | None,
    serial_decode_fallback: bool | None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for request_id in request_ids:
        ids = generated_ids[request_id]
        texts = generated_texts[request_id]
        prefix: list[int] = []
        for token_index, (token_id, token_text) in enumerate(zip(ids, texts, strict=True)):
            prefix.append(int(token_id))
            final = token_index == len(ids) - 1
            chunk = GenerationStreamChunk(
                text=token_text,
                finish_details=(
                    _finish_details_for_tokens(
                        tokenizer,
                        prefix,
                        ignore_eos=ignore_eos,
                        stop_token_ids=stop_token_ids,
                        stop_token_sequences=stop_token_sequences,
                        max_tokens=max_tokens,
                        sampler_mode=sampler_mode,
                    )
                    if final
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_rows_by_request[request_id],
                    prefix,
                    row_index=request_id,
                    request_id=str(request_id),
                    sampler_mode=sampler_mode,
                    stop_token_sequences=stop_token_sequences,
                    phase="answer",
                    execution_path=execution_path,
                    native_compact_prefill=native_compact_prefill,
                    native_caware_decode=native_caware_decode,
                    serial_decode_fallback=serial_decode_fallback,
                ),
            )
            chunks.append(_scheduler_token_chunk_payload(request_id, token_index, int(token_id), chunk))
    return chunks


def _sampled_batch_scheduler_token_chunks(
    request_ids: tuple[int, ...],
    prompt_rows_by_request: dict[int, list[int]],
    output_steps: dict[int, list[Qwen35ParoAutoregressiveStepResult]],
    sampling_state_step_snapshots: dict[int, list[RowSamplingState]],
    *,
    tokenizer: Any,
    vocab_size: Any | None,
    request: GenerationRequest,
    plans: dict[int, Any],
    execution_path: str | None,
    native_compact_prefill: bool | None,
    native_caware_decode: bool | None,
    serial_decode_fallback: bool | None,
    native_sampler_rows: bool | None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for request_id in request_ids:
        steps = output_steps[request_id]
        snapshots = sampling_state_step_snapshots[request_id]
        if len(snapshots) != len(steps):
            raise RuntimeError("sampled scheduler token snapshot count does not match generated steps")
        plan = plans[request_id]
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            plan,
            vocab_size=vocab_size,
        )
        prefix: list[int] = []
        for token_index, (step, state) in enumerate(zip(steps, snapshots, strict=True)):
            prefix.append(int(step.token_id))
            final = token_index == len(steps) - 1
            phase = None if state.thinking_budget is not None else "answer"
            chunk = GenerationStreamChunk(
                text=step.token_text,
                token_logprobs=_stream_token_logprobs_from_step(tokenizer, step, request),
                finish_details=(
                    _finish_details_for_tokens(
                        tokenizer,
                        prefix,
                        ignore_eos=request.ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=request.max_tokens,
                        sampler_mode=plan.mode.value,
                        sampling_state=state,
                    )
                    if final
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_rows_by_request[request_id],
                    prefix,
                    row_index=request_id,
                    request_id=str(request_id),
                    sampler_mode=plan.mode.value,
                    stop_token_sequences=request.stop_token_sequences,
                    phase=phase,
                    active_processors=plan.active_processors,
                    sampler_fast_path_blockers=plan.fast_path_blockers,
                    sampler_fallback_reason=plan.fallback_reason,
                    sampling_state=state,
                    forced_sample=step,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                    execution_path=execution_path,
                    native_compact_prefill=native_compact_prefill,
                    native_caware_decode=native_caware_decode,
                    serial_decode_fallback=serial_decode_fallback,
                    native_sampler_rows=native_sampler_rows,
                ),
            )
            chunks.append(_scheduler_token_chunk_payload(request_id, token_index, int(step.token_id), chunk))
    return chunks


def _scheduler_token_chunk_payload(
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
        "chunk": {
            "text": chunk.text,
        },
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


def _stream_token_logprobs_from_step(
    tokenizer: Any,
    step: Qwen35ParoAutoregressiveStepResult,
    request: GenerationRequest,
) -> tuple[TokenLogprob, ...]:
    if not request.logprobs and int(request.top_logprobs) <= 0:
        return ()
    return (_token_logprob_from_step(tokenizer, step),)


def _token_logprob_from_step(tokenizer: Any, step: Qwen35ParoAutoregressiveStepResult) -> TokenLogprob:
    return TokenLogprob(
        token_id=step.token_id,
        token_text=step.token_text,
        logprob=step.logprob,
        top_logprobs=tuple(
            (token_id, _decode_token_cached(tokenizer, token_id), logprob)
            for token_id, logprob in step.top_logprobs
        ),
    )


def _telemetry_for_tokens(
    prompt_ids: list[int] | tuple[int, ...],
    generated_token_ids: list[int] | tuple[int, ...],
    *,
    row_index: int,
    sampler_mode: str,
    stop_token_sequences: tuple[tuple[int, ...], ...],
    request_id: str | None = None,
    phase: str | None = None,
    active_processors: tuple[str, ...] = (),
    sampler_fast_path_blockers: tuple[str, ...] = (),
    sampler_fallback_reason: str | None = None,
    sampling_state: RowSamplingState | None = None,
    forced_sample: Qwen35ParoAutoregressiveStepResult | None = None,
    full_vocab_logits_d2h: bool | None = None,
    logits_d2h_bytes: int | None = None,
    execution_path: str | None = None,
    native_compact_prefill: bool | None = None,
    native_caware_decode: bool | None = None,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool | None = None,
) -> GenerationTelemetry:
    state_payload = _decode_state_from_sampling_state(sampling_state)
    forced_token_id, forced_token_reason, forced_tokens_remaining = _forced_token_metadata(forced_sample)
    return GenerationTelemetry.from_decode_counts(
        request_id=request_id,
        row_index=row_index,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_token_ids),
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
        sampler_mode=sampler_mode,
        stop_suffix_state=_stop_suffix_state(generated_token_ids, stop_token_sequences),
        active_processors=active_processors,
        sampler_fast_path_blockers=sampler_fast_path_blockers,
        sampler_fallback_reason=sampler_fallback_reason,
        full_vocab_logits_d2h=full_vocab_logits_d2h,
        logits_d2h_bytes=logits_d2h_bytes,
        execution_path=execution_path,
        native_compact_prefill=native_compact_prefill,
        native_caware_decode=native_caware_decode,
        serial_decode_fallback=serial_decode_fallback,
        native_sampler_rows=native_sampler_rows,
    )


def _forced_token_metadata(
    sample: Qwen35ParoAutoregressiveStepResult | None,
) -> tuple[int | None, str | None, int | None]:
    if sample is None or not bool(getattr(sample, "forced", False)):
        return None, None, None
    return (
        int(sample.token_id),
        None if sample.forced_reason is None else str(sample.forced_reason),
        max(0, int(sample.forced_tokens_remaining)),
    )


def _sampler_logits_d2h_metadata(
    plan: Any,
    *,
    vocab_size: Any | None = None,
) -> tuple[bool | None, int | None]:
    mode = getattr(plan, "mode", None)
    if mode is SamplingMode.GPU_SAMPLE:
        return False, 0
    if mode in (SamplingMode.HOST_LOGITS_SAMPLE, SamplingMode.PROCESSED_ARGMAX):
        try:
            size = int(vocab_size)
        except (TypeError, ValueError):
            return None, None
        if size > 0:
            return True, size * 4
    return None, None


def _decode_state_from_sampling_state(state: RowSamplingState | None) -> dict[str, Any]:
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


def _stop_suffix_state(
    generated_token_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> dict[str, Any] | None:
    payload = token_sequence_state_for_tokens(generated_token_ids, stop_token_sequences).to_json_dict()
    return payload or None


def _row_sampling_state(
    request: GenerationRequest,
    prompt_ids: list[int] | tuple[int, ...],
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


def _configure_sampled_session(
    session: Any,
    request: GenerationRequest | None,
    state: RowSamplingState | None,
    *,
    plan,
) -> None:
    if plan.mode is SamplingMode.GPU_SAMPLE:
        _configure_native_sampler(session, request, state)
    else:
        _configure_host_sampler(session, request, state)


def _configure_native_sampler(
    session: Any,
    request: GenerationRequest | None,
    state: RowSamplingState | None,
) -> None:
    configure = getattr(session, "configure_native_sampler", None)
    if not callable(configure):
        if request is None and state is None:
            return
        raise NotImplementedError(
            "Qwen3.5/PARO native GPU sampling requires resident sampler support"
        )
    configure(request, state)


def _configure_host_sampler(
    session: Any,
    request: GenerationRequest | None,
    state: RowSamplingState | None,
) -> None:
    configure = getattr(session, "configure_host_sampler", None)
    if not callable(configure):
        if request is None and state is None:
            return
        raise NotImplementedError(
            "Qwen3.5/PARO host-logits sampling requires resident sampler support"
        )
    configure(request, state)


def _native_gpu_sampler_route_available(*, prompt_count: int) -> bool:
    return int(prompt_count) == 1 and _native_gpu_sampler_requested()


def _native_gpu_sampler_requested() -> bool:
    return _env_flag("HIPENGINE_QWEN35_NATIVE_SAMPLER", default=True)


def _session_capacity_for(required_sequence_length: int) -> int:
    """Return a reusable session capacity for a request.

    Chat prompts grow after every turn, so allocating exactly the current
    prompt+decode length forces resident weights/KV buffers to be torn down and
    rebuilt on each request.  Keep a modest floor and bucket growth to preserve
    the resident session across normal local chat turns while still allowing
    larger explicit contexts to expand on demand.
    """

    required = int(required_sequence_length)
    if required <= 0:
        raise ValueError("required_sequence_length must be positive")
    floor = max(1, _env_int("HIPENGINE_SESSION_MIN_TOKENS", 4096))
    bucket = max(1, _env_int("HIPENGINE_SESSION_BUCKET_TOKENS", 1024))
    capacity = max(required, floor)
    return ((capacity + bucket - 1) // bucket) * bucket


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_eos(tokenizer: Any | None, token_id: int) -> bool:
    eos_id = _tokenizer_eos_id(tokenizer)
    return eos_id is not None and int(token_id) == int(eos_id)


def _tokenizer_eos_id(tokenizer: Any | None) -> int | None:
    if tokenizer is None:
        return None
    try:
        token_to_id = getattr(tokenizer, "token_to_id")
        eos_id = token_to_id("<|endoftext|>")
    except Exception:
        eos_id = None
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
    return None if eos_id is None else int(eos_id)


def _request_with_tokenizer_eos(
    request: GenerationRequest,
    tokenizer: Any | None,
) -> GenerationRequest:
    if request.eos_token_id is not None:
        return request
    eos_token_id = _tokenizer_eos_id(tokenizer)
    if eos_token_id is None:
        return request
    return replace(request, eos_token_id=eos_token_id)


def _queue_json_object_close_if_needed(
    state: RowSamplingState,
    tokenizer: Any | None,
    token_text: str,
    *,
    remaining_tokens: int,
) -> None:
    state.observe_text_for_json_object_close(
        token_text,
        remaining_tokens=remaining_tokens,
        encode_text=lambda text: _tokenize_constraint_text(tokenizer, text),
    )


def _tokenize_constraint_text(tokenizer: Any | None, text: str) -> tuple[int, ...]:
    if tokenizer is None:
        return ()
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            token_ids = tuple(int(token) for token in encode(str(text)))
        except Exception:
            token_ids = ()
        if token_ids:
            return token_ids
    token_to_id = getattr(tokenizer, "token_to_id", None)
    whole = _lookup_token_id(token_to_id, str(text))
    if whole is not None:
        return (whole,)
    pieces: list[int] = []
    for char in str(text):
        token_id = _lookup_token_id(token_to_id, char)
        if token_id is None:
            return ()
        pieces.append(token_id)
    return tuple(pieces)


def _lookup_token_id(token_to_id: Any, token: str) -> int | None:
    try:
        value = token_to_id(token) if callable(token_to_id) else token_to_id.get(token)
    except Exception:
        return None
    return None if value is None else int(value)


def _is_finished(
    tokenizer: Any | None,
    generated_token_ids: list[int] | tuple[int, ...],
    *,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> bool:
    if not generated_token_ids:
        return False
    token_id = int(generated_token_ids[-1])
    if not ignore_eos and _is_eos(tokenizer, token_id):
        return True
    if token_id in {int(stop_id) for stop_id in stop_token_ids}:
        return True
    return _ends_with_stop_sequence(generated_token_ids, stop_token_sequences)


def _finish_details_for_tokens(
    tokenizer: Any | None,
    generated_token_ids: list[int] | tuple[int, ...],
    *,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
    max_tokens: int,
    sampler_mode: str,
    sampling_state: RowSamplingState | None = None,
) -> FinishDetails:
    details: FinishDetails
    if generated_token_ids:
        token_id = int(generated_token_ids[-1])
        if not ignore_eos and _is_eos(tokenizer, token_id):
            details = FinishDetails(reason="eos", eos_token_id=token_id, sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, sampling_state)
        if token_id in {int(stop_id) for stop_id in stop_token_ids}:
            details = FinishDetails(reason="stop", stop_sequence=(token_id,), sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, sampling_state)
        sequence = _matched_stop_sequence(generated_token_ids, stop_token_sequences)
        if sequence:
            details = FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, sampling_state)
    if len(generated_token_ids) >= max(0, int(max_tokens)):
        details = FinishDetails(reason="length", length_limit=max_tokens, sampler_mode=sampler_mode)
        return finish_details_with_sampling_state(details, sampling_state)
    details = FinishDetails(reason="stop", sampler_mode=sampler_mode)
    return finish_details_with_sampling_state(details, sampling_state)


def _matched_stop_sequence(
    generated_token_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return token_sequence_state_for_tokens(generated_token_ids, stop_token_sequences).matched_sequence


def _ends_with_stop_sequence(
    generated_token_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> bool:
    return token_sequence_state_for_tokens(generated_token_ids, stop_token_sequences).matched


def make_qwen35_paro_one_token_generator(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
    )


def make_qwen35_paro_one_token_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1100",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator,
)
register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1151",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator_gfx1151,
)
