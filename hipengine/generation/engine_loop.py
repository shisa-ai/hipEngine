"""Long-lived scheduler-owned generation loop scaffolding.

This module is intentionally host-only and torch-free.  It wires the existing
``ResidentBatchScheduler`` to a small runner protocol so tests and early server
adapters can exercise a persistent ``submit``/``poll``/``cancel`` lifecycle
before native c>N sessions become correctness-green.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Protocol, Sequence

from hipengine.dispatch import RequestState, SlotMove, WorkItem, WorkKind
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken, ResidentBatchScheduler
from hipengine.generation.deadline import GenerationCancelled
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    TextGenerator,
)

PREFILL_DECODE_POLICIES = ("protect_decode", "protect_ttft", "fair")
DEFAULT_KV_POOL_INITIAL_PAGES = 128
DEFAULT_KV_POOL_LOW_WATER_PAGES = 128
DEFAULT_KV_POOL_CHUNK_PAGES = 128
DEFAULT_KV_POOL_IDLE_GRACE_SECONDS = 30.0
DEFAULT_MAX_PREFILL_CHUNK_TOKENS = 256
DEFAULT_RESIDENT_STREAM_QUEUE_MAX_CHUNKS = 64


@dataclass(frozen=True, slots=True)
class EngineLoopConfig:
    """CLI/env-resolved knobs for the C4 scheduler-owned engine loop."""

    prefill_decode_policy: str = "protect_decode"
    max_active_requests: int | None = None
    max_prefill_chunk_tokens: int = DEFAULT_MAX_PREFILL_CHUNK_TOKENS
    kv_pool_initial_pages: int = DEFAULT_KV_POOL_INITIAL_PAGES
    kv_pool_low_water_pages: int = DEFAULT_KV_POOL_LOW_WATER_PAGES
    kv_pool_high_water_pages: int | None = None
    kv_pool_chunk_pages: int = DEFAULT_KV_POOL_CHUNK_PAGES
    kv_pool_idle_grace_seconds: float = DEFAULT_KV_POOL_IDLE_GRACE_SECONDS
    max_pending_requests: int | None = None

    def __post_init__(self) -> None:
        if self.prefill_decode_policy not in PREFILL_DECODE_POLICIES:
            raise ValueError(f"prefill_decode_policy must be one of {PREFILL_DECODE_POLICIES!r}")
        if self.max_active_requests is not None and self.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive when set")
        if self.max_prefill_chunk_tokens <= 0:
            raise ValueError("max_prefill_chunk_tokens must be positive")
        if self.kv_pool_initial_pages <= 0:
            raise ValueError("kv_pool_initial_pages must be positive")
        if self.kv_pool_low_water_pages <= 0:
            raise ValueError("kv_pool_low_water_pages must be positive")
        if self.kv_pool_low_water_pages > self.kv_pool_initial_pages:
            raise ValueError("kv_pool_low_water_pages cannot exceed kv_pool_initial_pages")
        if self.kv_pool_high_water_pages is not None and self.kv_pool_high_water_pages < self.kv_pool_initial_pages:
            raise ValueError("kv_pool_high_water_pages cannot be below kv_pool_initial_pages")
        if self.kv_pool_chunk_pages <= 0:
            raise ValueError("kv_pool_chunk_pages must be positive")
        if self.kv_pool_idle_grace_seconds < 0:
            raise ValueError("kv_pool_idle_grace_seconds must be non-negative")
        if self.max_pending_requests is not None and self.max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be positive when set")


@dataclass(frozen=True, slots=True)
class GenerationSubmission:
    """Stable request ids for one batch submitted to a shared model loop."""

    request_ids: tuple[int, ...]
    request: GenerationRequest
    max_ticks: int


@dataclass(slots=True)
class _ResidentStreamState:
    submission: GenerationSubmission
    events: deque[tuple[int, GenerationStreamChunk]] = field(default_factory=deque)
    emitted_text_request_ids: set[int] = field(default_factory=set)
    overflowed_request_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class EngineLoopEvent:
    """One externally visible event produced by ``ResidentEngineLoop.poll``."""

    kind: str
    request_id: int | None = None
    request_ids: tuple[int, ...] = ()
    work_kind: WorkKind | None = None
    token_id: int | None = None
    stream_chunk: GenerationStreamChunk | None = None
    completed: CompletedRequest | None = None


class EngineLoopRunner(Protocol):
    """Scheduler-facing commit/compact/reclaim model-runner contract.

    Native runners implement the batch-shaped methods.  The historical
    ``prefill``/``decode`` pair remains an explicit serial bridge for simple
    host fakes and non-native generators.
    """

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        """Run one prefill transition and commit canonical state when requested."""

    def decode_batch(self, work: WorkItem, *, commit: bool) -> Sequence[GeneratedToken]:
        """Return one generated token per decoded request row."""

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        """Apply scheduler physical-slot moves to model-owned state."""

    def reclaim(self, completed: CompletedRequest) -> None:
        """Release model-owned state for one completed request."""

    def reserve_admission(self, request: RequestState) -> None:
        """Reserve model/KV resources before scheduler slot publication."""

    def rollback_admission(self, request: RequestState) -> None:
        """Release an unpublished reservation after slot-commit failure."""


class SubmitPollTextGenerator:
    """Run a ``TextGenerator`` through the resident ``submit``/``poll`` loop.

    The wrapped generator still owns tokenization and model execution.  This
    adapter gives public ``LLM.generate()`` the same request-id preserving
    lifecycle as the C4 resident scheduler: rows are submitted, prefilled by
    scheduler work items, decoded as one text batch, and collected by completion
    request id.  It is a host-side serial bridge until native token streaming
    runners replace the inner ``generate`` call.
    """

    def __init__(
        self,
        inner: TextGenerator,
        *,
        capacity: int | None = None,
        prefill_chunk_size: int = 1024,
        context_bucket_size: int = 256,
        config: EngineLoopConfig | None = None,
        stream_queue_max_chunks: int = DEFAULT_RESIDENT_STREAM_QUEUE_MAX_CHUNKS,
    ) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if stream_queue_max_chunks <= 0:
            raise ValueError("stream_queue_max_chunks must be positive")
        if (
            config is not None
            and capacity is not None
            and config.max_active_requests is not None
            and int(config.max_active_requests) != int(capacity)
        ):
            raise ValueError("capacity conflicts with config.max_active_requests")
        self._inner = inner
        self._context_bucket_size = int(context_bucket_size)
        configured_capacity = (
            capacity
            if capacity is not None
            else (None if config is None else config.max_active_requests)
        )
        resident_runner_factory = getattr(self._inner, "create_resident_model_runner", None)
        has_resident_runner = callable(resident_runner_factory)
        self._has_resident_runner = bool(has_resident_runner)
        if has_resident_runner:
            self._runner = resident_runner_factory(capacity=configured_capacity)
            resolved_capacity = int(self._runner.capacity)
        else:
            resolved_capacity = 32 if configured_capacity is None else int(configured_capacity)
            self._runner = _SubmitPollTextRunner(self._inner, capacity=resolved_capacity)
        if config is None:
            self._loop = ResidentEngineLoop(
                self._runner,
                capacity=resolved_capacity,
                prefill_chunk_size=int(prefill_chunk_size),
                context_bucket_size=self._context_bucket_size,
                prefill_decode_policy="protect_ttft",
            )
        else:
            loop_config = (
                config
                if config.max_active_requests is not None
                else replace(config, max_active_requests=resolved_capacity)
            )
            if not has_resident_runner:
                # The compatibility bridge invokes one whole inner generation
                # call per decode work item.  Prefill every submitted row first
                # so prompt-list calls remain one ordered inner batch; native
                # resident runners consume the configured scheduling policy.
                loop_config = replace(loop_config, prefill_decode_policy="protect_ttft")
            self._loop = ResidentEngineLoop(
                self._runner,
                capacity=resolved_capacity,
                context_bucket_size=self._context_bucket_size,
                config=loop_config,
            )
        self._prefill_chunk_size = int(self._loop.prefill_chunk_size)
        configure_engine_loop = getattr(self._runner, "configure_engine_loop", None)
        if callable(configure_engine_loop):
            configure_engine_loop(self._loop.config)
        # The lock protects one mutable scheduler tick, never an entire request.
        # Native runners therefore release it after each model transition so a
        # later D2 admission worker can enqueue between decode steps.
        self._loop_lock = threading.Lock()
        self._stream_queue_max_chunks = int(stream_queue_max_chunks)
        self._stream_states_by_request: dict[int, _ResidentStreamState] = {}

    @property
    def inner(self) -> TextGenerator:
        return self._inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int | None:
        preparer = getattr(self._inner, "prepare", None)
        result = None
        if callable(preparer):
            result = preparer(
                max_sequence_length=max_sequence_length,
                sampling_params=sampling_params,
            )
        runner_preparer = getattr(self._runner, "prepare", None)
        if callable(runner_preparer):
            runner_preparer(max_sequence_length=max_sequence_length)
        return result

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        prompts = tuple(request.prompts)
        if not prompts:
            return []
        submission = self.submit_detailed(replace(request, prompts=prompts))
        ticks = 0
        try:
            while not self.generation_complete(submission):
                events = self.poll(max_ticks=1)
                ticks += 1
                if not events:
                    missing = self._runner.missing_outputs(submission.request_ids)
                    raise RuntimeError(f"submit+poll text generation stalled; missing request_ids={missing}")
                if ticks > submission.max_ticks:
                    missing = self._runner.missing_outputs(submission.request_ids)
                    raise RuntimeError(
                        f"submit+poll text generation exceeded {submission.max_ticks} ticks; "
                        f"missing request_ids={missing}"
                    )
            return self.take_result(submission)
        except Exception:
            self._abort_submission(submission)
            raise

    def submit_detailed(self, request: GenerationRequest) -> GenerationSubmission:
        """Submit rows without driving the shared loop to completion."""

        with self._loop_lock:
            return self._submit_detailed_locked(request)

    def _submit_detailed_locked(self, request: GenerationRequest) -> GenerationSubmission:
        prompts = tuple(request.prompts)
        if not prompts:
            raise ValueError("submit_detailed requires at least one prompt")
        normalized = replace(request, prompts=prompts)
        prompt_rows = tuple(self._runner.prompt_tokens(prompt) for prompt in prompts)
        max_new_tokens = int(self._runner.scheduler_max_new_tokens(normalized))
        request_ids: list[int] = []
        try:
            for prompt_row in prompt_rows:
                request_ids.append(
                    self._loop.submit(prompt_row, max_new_tokens=max_new_tokens)
                )
            self._runner.register_batch(
                request_ids,
                normalized,
                prompt_rows=prompt_rows,
            )
        except Exception:
            for request_id in request_ids:
                self._loop.cancel(request_id)
                self._loop.release_completed(request_id)
            self._runner.discard(request_ids)
            raise
        return GenerationSubmission(
            request_ids=tuple(request_ids),
            request=normalized,
            max_ticks=_submit_poll_max_ticks(
                prompt_rows,
                self._prefill_chunk_size,
                max_new_tokens=max_new_tokens,
            ),
        )

    def poll(self, *, max_ticks: int = 1) -> tuple[EngineLoopEvent, ...]:
        """Advance shared model work without owning a request-lifetime lock."""

        with self._loop_lock:
            events = self._loop.poll(max_ticks=max_ticks)
            self._route_stream_events_locked(events)
            return events

    def generation_complete(self, submission: GenerationSubmission) -> bool:
        with self._loop_lock:
            return self._runner.has_outputs(submission.request_ids)

    def cancel_submission(
        self,
        submission: GenerationSubmission,
        *,
        row_index: int | None = None,
        reason: str = "cancel",
    ) -> tuple[bool, ...]:
        """Cancel all rows or one local row through the unified reclaim path."""

        if row_index is None:
            request_ids = submission.request_ids
        else:
            index = int(row_index)
            if index < 0 or index >= len(submission.request_ids):
                raise IndexError("row_index is outside the submitted generation")
            request_ids = (submission.request_ids[index],)
        with self._loop_lock:
            return tuple(
                self._loop.cancel(request_id, reason=reason)
                for request_id in request_ids
            )

    def take_result(self, submission: GenerationSubmission) -> list[GenerationOutput]:
        """Consume one completed submission in original prompt order."""

        with self._loop_lock:
            if not self._runner.has_outputs(submission.request_ids):
                missing = self._runner.missing_outputs(submission.request_ids)
                raise RuntimeError(f"submitted generation is incomplete; missing request_ids={missing}")
            outputs = self._runner.take_outputs(submission.request_ids)
            for request_id in submission.request_ids:
                self._loop.release_completed(request_id)
            finalize_batch = getattr(self._runner, "finalize_batch", None)
            if callable(finalize_batch):
                finalize_batch(submission.request, submission.request_ids, outputs)
            return outputs

    def _abort_submission(self, submission: GenerationSubmission) -> None:
        with self._loop_lock:
            self._abort_submission_locked(submission, reason="cancel")

    def live_loop_snapshot(self) -> dict[str, object]:
        """Return one lock-consistent scheduler plus model-runner snapshot."""

        with self._loop_lock:
            runner_snapshot = getattr(self._runner, "observability_snapshot", None)
            return {
                "schema": 1,
                "kind": "resident_engine_loop_observability",
                "loop": self._loop.observability_snapshot(),
                "runner": (
                    runner_snapshot()
                    if callable(runner_snapshot)
                    else {}
                ),
            }

    @property
    def supports_stream_many(self) -> bool:
        return bool(
            self._has_resident_runner
            or getattr(self._inner, "supports_stream_many", False)
            or getattr(self._inner, "supports_stream_many_detailed", False)
        )

    @property
    def supports_controlled_streaming(self) -> bool:
        return self._has_resident_runner

    @property
    def stream_queue_max_chunks(self) -> int:
        return self._stream_queue_max_chunks

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield str(chunk)

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        if self._has_resident_runner:
            yield from self.stream_many_detailed(request)
            return
        detailed_streamer = getattr(self._inner, "stream_detailed", None)
        if callable(detailed_streamer):
            for chunk in detailed_streamer(request):
                yield GenerationStreamChunk.from_value(chunk)
            return
        streamer = getattr(self._inner, "stream", None)
        if callable(streamer):
            for chunk in streamer(request):
                yield GenerationStreamChunk.from_value(chunk)
            return
        for output in self.generate_detailed(request):
            generation_output = (
                output if isinstance(output, GenerationOutput) else GenerationOutput(text=str(output))
            )
            yield GenerationStreamChunk(
                text=generation_output.text,
                token_logprobs=generation_output.token_logprobs,
                finish_details=generation_output.finish_details,
                telemetry=generation_output.telemetry,
            )

    def stream_many_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        """Stream row-indexed events from the shared resident loop.

        Every iterator owns only its subscription. Any iterator may advance one
        scheduler tick; emitted events are routed to the matching subscription
        before the tick lock is released, so concurrent callers cannot consume
        one another's rows.
        """

        if not self._has_resident_runner:
            detailed_streamer = getattr(self._inner, "stream_many_detailed", None)
            if not callable(detailed_streamer):
                raise NotImplementedError("multi-row streaming is not supported by this generator")
            for chunk in detailed_streamer(request):
                yield GenerationStreamChunk.from_value(chunk)
            return

        with self._loop_lock:
            submission = self._submit_detailed_locked(request)
            state = _ResidentStreamState(submission)
            for request_id in submission.request_ids:
                if request_id in self._stream_states_by_request:
                    self._abort_submission_locked(submission, reason="cancel")
                    raise RuntimeError(f"request_id {request_id} already has a stream subscription")
                self._stream_states_by_request[request_id] = state

        consumed = False
        ticks = 0
        try:
            while True:
                queued: tuple[int, GenerationStreamChunk] | None = None
                with self._loop_lock:
                    if state.events:
                        queued = state.events.popleft()
                    complete = self._runner.has_outputs(submission.request_ids)
                if queued is not None:
                    request_id, chunk = queued
                    if chunk.text:
                        state.emitted_text_request_ids.add(request_id)
                    if chunk.text or chunk.token_logprobs:
                        yield chunk
                    continue
                if state.overflowed_request_ids:
                    raise GenerationCancelled(
                        FinishDetails(
                            reason="cancelled",
                            cancelled=True,
                            budget_pressure="client_backpressure",
                        )
                    )
                if complete:
                    break
                events = self.poll(max_ticks=1)
                ticks += 1
                if not events:
                    # Another stream may complete this subscription after the
                    # pre-poll check but before our shared-loop poll acquires
                    # the lock. Re-read routed events/output ownership before
                    # diagnosing a stall; an empty missing list is progress.
                    with self._loop_lock:
                        progressed = bool(state.events) or self._runner.has_outputs(
                            submission.request_ids
                        )
                    if progressed:
                        continue
                    missing = self._runner.missing_outputs(submission.request_ids)
                    raise RuntimeError(
                        f"resident stream stalled; missing request_ids={missing}"
                    )
                if ticks > submission.max_ticks:
                    missing = self._runner.missing_outputs(submission.request_ids)
                    raise RuntimeError(
                        f"resident stream exceeded {submission.max_ticks} ticks; "
                        f"missing request_ids={missing}"
                    )

            outputs = self.take_result(submission)
            with self._loop_lock:
                self._unregister_stream_state_locked(state)
            for request_id, output in zip(submission.request_ids, outputs, strict=True):
                generation_output = (
                    output
                    if isinstance(output, GenerationOutput)
                    else GenerationOutput(text=str(output))
                )
                if request_id in state.emitted_text_request_ids:
                    continue
                yield GenerationStreamChunk(
                    text=generation_output.text,
                    token_logprobs=generation_output.token_logprobs,
                    finish_details=generation_output.finish_details,
                    telemetry=generation_output.telemetry,
                )
            consumed = True
        finally:
            if not consumed:
                with self._loop_lock:
                    self._abort_submission_locked(submission, reason="disconnect")
                    self._unregister_stream_state_locked(state)

    def _route_stream_events_locked(self, events: Sequence[EngineLoopEvent]) -> None:
        for event in events:
            if event.kind != "token" or event.request_id is None or event.stream_chunk is None:
                continue
            request_id = int(event.request_id)
            state = self._stream_states_by_request.get(request_id)
            if state is None:
                continue
            if len(state.events) >= self._stream_queue_max_chunks:
                state.overflowed_request_ids.add(request_id)
                self._loop.cancel(request_id, reason="cancel")
                continue
            state.events.append((request_id, event.stream_chunk))

    def _unregister_stream_state_locked(self, state: _ResidentStreamState) -> None:
        for request_id in state.submission.request_ids:
            if self._stream_states_by_request.get(request_id) is state:
                self._stream_states_by_request.pop(request_id, None)

    def _abort_submission_locked(
        self,
        submission: GenerationSubmission,
        *,
        reason: str,
    ) -> None:
        for request_id in submission.request_ids:
            self._loop.cancel(request_id, reason=reason)
            self._loop.release_completed(request_id)
        self._runner.discard(submission.request_ids)

    def close(self) -> None:
        """Force-reclaim any remaining rows and release long-lived resources."""

        with self._loop_lock:
            states = {id(state): state for state in self._stream_states_by_request.values()}
            for state in states.values():
                self._abort_submission_locked(state.submission, reason="cancel")
            self._stream_states_by_request.clear()
            active_request_ids = tuple(getattr(self._runner, "active_request_ids", ()))
            for request_id in active_request_ids:
                self._loop.cancel(int(request_id), reason="cancel")
                self._loop.release_completed(int(request_id))
            if active_request_ids:
                self._runner.discard(active_request_ids)
            closer = getattr(self._runner, "close", None)
            if callable(closer):
                closer()


@dataclass(frozen=True, slots=True)
class _SubmitPollTextRow:
    batch_id: int
    row_index: int
    request: GenerationRequest


class _SubmitPollTextRunner:
    """Long-lived compatibility runner for non-native text generators."""

    def __init__(self, inner: TextGenerator, *, capacity: int) -> None:
        self._inner = inner
        self.capacity = int(capacity)
        self._rows: dict[int, _SubmitPollTextRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._next_batch_id = 0

    @property
    def outputs(self) -> dict[int, GenerationOutput]:
        return dict(self._outputs)

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    def prompt_tokens(self, prompt: Any) -> tuple[int, ...]:
        return _surrogate_prompt_tokens(prompt)

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        del request
        return 1

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        ids = tuple(int(request_id) for request_id in request_ids)
        if len(ids) != len(request.prompts):
            raise ValueError("request_ids must have one entry per prompt")
        if len(prompt_rows) != len(ids):
            raise ValueError("prompt_rows must have one entry per request_id")
        if request.row_seeds and len(request.row_seeds) != len(request.prompts):
            raise ValueError("row_seeds must have one entry per prompt")
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        for row_index, request_id in enumerate(ids):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            self._rows[request_id] = _SubmitPollTextRow(batch_id, row_index, request)

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("submit+poll compatibility prefill requires commit=True")

    def decode_batch(self, work: WorkItem, *, commit: bool) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("submit+poll compatibility decode requires commit=True")
        request_ids = tuple(int(request_id) for request_id in work.request_ids)
        grouped: dict[int, list[int]] = {}
        for request_id in request_ids:
            row = self._rows.get(request_id)
            if row is None:
                raise KeyError(f"request_id {request_id} is not registered")
            grouped.setdefault(row.batch_id, []).append(request_id)

        tokens_by_request: dict[int, GeneratedToken] = {}
        for grouped_ids in grouped.values():
            ids = tuple(grouped_ids)
            subrequest = self._subset_request(ids)
            detailed = getattr(self._inner, "generate_detailed", None)
            if callable(detailed):
                outputs = list(detailed(subrequest))
            else:
                outputs = [GenerationOutput(text=str(item)) for item in self._inner.generate(subrequest)]
            if len(outputs) != len(ids):
                raise RuntimeError(
                    f"generator returned {len(outputs)} outputs for {len(ids)} submit+poll rows"
                )
            for row_index, (request_id, output) in enumerate(zip(ids, outputs, strict=True)):
                generation_output = (
                    output if isinstance(output, GenerationOutput) else GenerationOutput(text=str(output))
                )
                self._outputs[request_id] = generation_output
                tokens_by_request[request_id] = GeneratedToken(request_id, row_index, finished=True)
        return tuple(tokens_by_request[request_id] for request_id in request_ids)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        del moves

    def reclaim(self, completed: CompletedRequest) -> None:
        # A compatibility decode normally publishes its complete output before
        # scheduler reclaim. Pending cancellation instead needs an explicit
        # empty result so a controlled submission can be consumed normally.
        if completed.request_id not in self._outputs and completed.request_id in self._rows:
            self._outputs[completed.request_id] = GenerationOutput(
                text="",
                generated_token_ids=completed.generated_tokens,
                finish_details=completed.finish_details,
            )

    def has_outputs(self, request_ids: Sequence[int]) -> bool:
        return all(int(request_id) in self._outputs for request_id in request_ids)

    def missing_outputs(self, request_ids: Sequence[int]) -> list[int]:
        return [int(request_id) for request_id in request_ids if int(request_id) not in self._outputs]

    def take_outputs(self, request_ids: Sequence[int]) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        for request_id in request_ids:
            rid = int(request_id)
            outputs.append(self._outputs.pop(rid))
            self._rows.pop(rid, None)
        return outputs

    def discard(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            rid = int(request_id)
            self._rows.pop(rid, None)
            self._outputs.pop(rid, None)

    def _subset_request(self, request_ids: tuple[int, ...]) -> GenerationRequest:
        rows = tuple(self._rows[request_id] for request_id in request_ids)
        request = rows[0].request
        if any(row.request is not request for row in rows):
            raise RuntimeError("submit+poll batch rows do not share one source request")
        prompts = tuple(request.prompts[row.row_index] for row in rows)
        row_seeds: tuple[int, ...] = ()
        if request.row_seeds:
            row_seeds = tuple(request.row_seeds[row.row_index] for row in rows)
        return replace(request, prompts=prompts, row_seeds=row_seeds)


def _surrogate_prompt_tokens(prompt: Any) -> tuple[int, ...]:
    # The inner text generator performs real tokenization.  The scheduler only
    # needs a non-empty non-negative row to exercise admission/prefill lifecycle.
    if isinstance(prompt, str):
        return (len(prompt.encode("utf-8")),)
    return (len(prompt),)


def _submit_poll_max_ticks(
    prompt_rows: Sequence[Sequence[int]],
    prefill_chunk_size: int,
    *,
    max_new_tokens: int,
) -> int:
    """Return a finite bound that covers real chunked prefill and token decode."""

    chunk_size = int(prefill_chunk_size)
    if chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    prefill_ticks = sum(
        max(1, (len(row) + chunk_size - 1) // chunk_size)
        for row in prompt_rows
    )
    # Decode advances every ready row once per tick, so the longest request—not
    # the sum across rows—sets the decode bound. Admission and reclaim events
    # share those work ticks; retain a small diagnostic margin for cancellation
    # or a late concurrent admission without allowing an infinite server loop.
    return max(8, prefill_ticks + max(1, int(max_new_tokens)) + len(prompt_rows) + 4)


def add_engine_loop_config_args(
    parser: argparse.ArgumentParser,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Add C4 engine-loop CLI knobs with env-backed defaults."""

    env = os.environ if environ is None else environ
    parser.add_argument(
        "--prefill-decode-policy",
        choices=PREFILL_DECODE_POLICIES,
        default=_env_prefill_decode_policy(env),
        help="Prefill/decode scheduler policy (env HIPENGINE_PREFILL_DECODE_POLICY; default: protect_decode)",
    )
    parser.add_argument(
        "--max-active-requests",
        type=_positive_int_arg,
        default=_env_optional_positive_int(env, "HIPENGINE_MAX_ACTIVE_REQUESTS"),
        help="Optional active resident request cap (env HIPENGINE_MAX_ACTIVE_REQUESTS; default: unset)",
    )
    parser.add_argument(
        "--max-prefill-chunk-tokens",
        type=_positive_int_arg,
        default=_env_positive_int(
            env,
            "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS",
            DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
        ),
        help="Maximum prefill chunk tokens per loop tick (env HIPENGINE_MAX_PREFILL_CHUNK_TOKENS; default: 256)",
    )
    parser.add_argument(
        "--kv-pool-initial-pages",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_KV_POOL_INITIAL_PAGES", DEFAULT_KV_POOL_INITIAL_PAGES),
        help="Initial dynamic KV pool pages (env HIPENGINE_KV_POOL_INITIAL_PAGES; default: 128)",
    )
    parser.add_argument(
        "--kv-pool-low-water-pages",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_KV_POOL_LOW_WATER_PAGES", DEFAULT_KV_POOL_LOW_WATER_PAGES),
        help="KV pool idle-shrink low-water pages (env HIPENGINE_KV_POOL_LOW_WATER_PAGES; default: 128)",
    )
    parser.add_argument(
        "--kv-pool-high-water-pages",
        type=_positive_int_arg,
        default=_env_optional_positive_int(env, "HIPENGINE_KV_POOL_HIGH_WATER_PAGES"),
        help="Optional KV pool high-water page cap (env HIPENGINE_KV_POOL_HIGH_WATER_PAGES; default: unset)",
    )
    parser.add_argument(
        "--kv-pool-chunk-pages",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_KV_POOL_CHUNK_PAGES", DEFAULT_KV_POOL_CHUNK_PAGES),
        help="KV pool grow/shrink chunk size in pages (env HIPENGINE_KV_POOL_CHUNK_PAGES; default: 128)",
    )
    parser.add_argument(
        "--kv-pool-idle-grace-seconds",
        type=_nonnegative_float_arg,
        default=_env_nonnegative_float(
            env,
            "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS",
            DEFAULT_KV_POOL_IDLE_GRACE_SECONDS,
        ),
        help="Seconds before idle tail chunks can shrink (env HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS; default: 30.0)",
    )
    parser.add_argument(
        "--max-pending-requests",
        type=_positive_int_arg,
        default=_env_optional_positive_int(env, "HIPENGINE_MAX_PENDING_REQUESTS"),
        help="Optional pending request queue cap (env HIPENGINE_MAX_PENDING_REQUESTS; default: unset)",
    )


def engine_loop_config_from_args(args: object) -> EngineLoopConfig:
    """Build an ``EngineLoopConfig`` from an argparse namespace-like object."""

    return EngineLoopConfig(
        prefill_decode_policy=str(getattr(args, "prefill_decode_policy")),
        max_active_requests=(
            None
            if getattr(args, "max_active_requests") is None
            else int(getattr(args, "max_active_requests"))
        ),
        max_prefill_chunk_tokens=int(getattr(args, "max_prefill_chunk_tokens")),
        kv_pool_initial_pages=int(getattr(args, "kv_pool_initial_pages")),
        kv_pool_low_water_pages=int(getattr(args, "kv_pool_low_water_pages")),
        kv_pool_high_water_pages=(
            None
            if getattr(args, "kv_pool_high_water_pages") is None
            else int(getattr(args, "kv_pool_high_water_pages"))
        ),
        kv_pool_chunk_pages=int(getattr(args, "kv_pool_chunk_pages")),
        kv_pool_idle_grace_seconds=float(getattr(args, "kv_pool_idle_grace_seconds")),
        max_pending_requests=(
            None
            if getattr(args, "max_pending_requests") is None
            else int(getattr(args, "max_pending_requests"))
        ),
    )


def engine_loop_config_from_env(environ: Mapping[str, str] | None = None) -> EngineLoopConfig:
    """Resolve C4 engine-loop knobs directly from environment values."""

    parser = argparse.ArgumentParser(add_help=False)
    add_engine_loop_config_args(parser, environ=environ)
    return engine_loop_config_from_args(parser.parse_args([]))


def _env_prefill_decode_policy(environ: Mapping[str, str]) -> str:
    raw = environ.get("HIPENGINE_PREFILL_DECODE_POLICY")
    value = "protect_decode" if raw is None or raw == "" else raw.strip()
    if value not in PREFILL_DECODE_POLICIES:
        raise ValueError(f"HIPENGINE_PREFILL_DECODE_POLICY must be one of {PREFILL_DECODE_POLICIES!r}")
    return value


def _env_positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    return int(default) if raw is None or raw == "" else _positive_int_arg(raw)


def _env_optional_positive_int(environ: Mapping[str, str], name: str) -> int | None:
    raw = environ.get(name)
    return None if raw is None or raw == "" else _positive_int_arg(raw)


def _env_nonnegative_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    return float(default) if raw is None or raw == "" else _nonnegative_float_arg(raw)


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


class ResidentEngineLoop:
    """Persistent ``submit``/``poll``/``cancel`` driver for resident batches.

    The loop currently executes at most one scheduler work item per tick.  It is
    deliberately conservative: requests stay resident across polls, admission
    fills free slots, the prefill/decode choice is explicit, and completion
    reclaim is delegated to ``ResidentBatchScheduler``.
    """

    def __init__(
        self,
        runner: EngineLoopRunner,
        *,
        capacity: int | None = None,
        prefill_chunk_size: int | None = None,
        context_bucket_size: int = 256,
        prefill_decode_policy: str = "protect_decode",
        max_pending_requests: int | None = None,
        config: EngineLoopConfig | None = None,
    ) -> None:
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        direct_override_with_config = (
            prefill_decode_policy != "protect_decode"
            or max_pending_requests is not None
            or prefill_chunk_size is not None
        )
        if config is not None and direct_override_with_config:
            raise ValueError("pass either config or direct engine-loop overrides, not both")
        if config is None:
            if capacity is None:
                raise ValueError("capacity is required")
            resolved_config = EngineLoopConfig(
                prefill_decode_policy=prefill_decode_policy,
                max_active_requests=int(capacity),
                max_prefill_chunk_tokens=(
                    DEFAULT_MAX_PREFILL_CHUNK_TOKENS
                    if prefill_chunk_size is None
                    else int(prefill_chunk_size)
                ),
                max_pending_requests=max_pending_requests,
            )
            resolved_capacity = resolved_config.max_active_requests
        else:
            resolved_config = config
            if capacity is None:
                if resolved_config.max_active_requests is None:
                    raise ValueError("capacity or config.max_active_requests is required")
                resolved_capacity = resolved_config.max_active_requests
            else:
                resolved_capacity = int(capacity)
                if resolved_capacity <= 0:
                    raise ValueError("capacity must be positive")
                if (
                    resolved_config.max_active_requests is not None
                    and resolved_config.max_active_requests != resolved_capacity
                ):
                    raise ValueError("capacity conflicts with config.max_active_requests")
        assert resolved_capacity is not None
        self.runner = runner
        self.prefill_chunk_size = int(resolved_config.max_prefill_chunk_tokens)
        self.config = resolved_config
        self.prefill_decode_policy = resolved_config.prefill_decode_policy
        self._last_work_kind: WorkKind | None = None
        self.scheduler = ResidentBatchScheduler(
            capacity=resolved_capacity,
            context_bucket_size=context_bucket_size,
            max_pending_requests=resolved_config.max_pending_requests,
            reclaim_callback=self._reclaim_runner_state,
        )

    @property
    def pending_count(self) -> int:
        return self.scheduler.pending_count

    @property
    def active_count(self) -> int:
        return self.scheduler.active_count

    @property
    def completed(self) -> dict[int, CompletedRequest]:
        return dict(self.scheduler.completed)

    def observability_snapshot(self) -> dict[str, object]:
        """Return scheduler ownership, work, policy, and latency evidence."""

        snapshot = self.scheduler.observability_snapshot()
        snapshot["scheduler_policy"] = {
            "prefill_decode_policy": self.prefill_decode_policy,
            "prefill_chunk_tokens": int(self.prefill_chunk_size),
            "last_work_kind": (
                None if self._last_work_kind is None else self._last_work_kind.value
            ),
        }
        return snapshot

    def submit(self, prompt_tokens: Iterable[int], *, max_new_tokens: int, request_id: int | None = None) -> int:
        return self.scheduler.submit(prompt_tokens, max_new_tokens=max_new_tokens, request_id=request_id)

    def cancel(self, request_id: int, *, reason: str = "cancel") -> bool:
        """Cancel a pending or active request and reclaim active scheduler state."""

        return self.scheduler.cancel(request_id, reason=reason) is not None

    def compact(self, order: Sequence[int] | None = None) -> tuple[SlotMove, ...]:
        """Compact scheduler slots and commit the same moves to the runner."""

        moves = tuple(self.scheduler.compact(order=order))
        compact_batch = getattr(self.runner, "compact_batch", None)
        if callable(compact_batch):
            compact_batch(moves)
        return moves

    def release_completed(self, request_id: int) -> CompletedRequest | None:
        """Release one caller-consumed completion from the long-lived loop."""

        return self.scheduler.release_completed(request_id)

    def disconnect(self, request_id: int) -> bool:
        """Reclaim a disconnected request through the unified cancel path."""

        return self.cancel(request_id, reason="disconnect")

    def timeout(self, request_id: int) -> bool:
        """Reclaim a timed-out request through the unified cancel path."""

        return self.cancel(request_id, reason="timeout")

    def poll(self, *, max_ticks: int = 1) -> tuple[EngineLoopEvent, ...]:
        """Advance the loop by up to ``max_ticks`` scheduler ticks."""

        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        events: list[EngineLoopEvent] = []
        for _ in range(int(max_ticks)):
            tick_events = self.tick()
            if not tick_events:
                break
            events.extend(tick_events)
        return tuple(events)

    def tick(self) -> tuple[EngineLoopEvent, ...]:
        """Run one admission/prefill/decode tick and finish at a maintenance barrier."""

        try:
            return self._tick_once()
        finally:
            barrier = getattr(self.runner, "loop_barrier", None)
            if callable(barrier):
                barrier(active_count=self.active_count, pending_count=self.pending_count)

    def _tick_once(self) -> tuple[EngineLoopEvent, ...]:
        events: list[EngineLoopEvent] = []
        reserve_admission = getattr(self.runner, "reserve_admission", None)
        rollback_admission = getattr(self.runner, "rollback_admission", None)
        admitted = self.scheduler.admit_pending(
            reserve_callback=(reserve_admission if callable(reserve_admission) else None),
            rollback_callback=(rollback_admission if callable(rollback_admission) else None),
        )
        events.extend(
            EngineLoopEvent(kind="admitted", request_id=request_id, request_ids=(request_id,))
            for request_id in admitted
        )

        decode = self.scheduler.next_decode_work()
        prefill_available = self.scheduler.has_prefill_work()
        if self._choose_decode_first(decode_available=decode is not None, prefill_available=prefill_available):
            assert decode is not None
            events.extend(self._run_decode(decode))
            return tuple(events)

        if prefill_available:
            prefill = self.scheduler.next_prefill_work(chunk_size=self.prefill_chunk_size)
            assert prefill is not None
            events.extend(self._run_prefill(prefill))
            return tuple(events)

        if decode is None:
            return tuple(events)
        events.extend(self._run_decode(decode))
        return tuple(events)

    def _choose_decode_first(self, *, decode_available: bool, prefill_available: bool) -> bool:
        if not decode_available:
            return False
        if not prefill_available:
            return True
        if self.prefill_decode_policy == "protect_decode":
            return True
        if self.prefill_decode_policy == "protect_ttft":
            return False
        return self._last_work_kind is WorkKind.PREFILL

    def _run_prefill(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        start = time.perf_counter()
        prefill_batch = getattr(self.runner, "prefill_batch", None)
        if callable(prefill_batch):
            prefill_batch(work, commit=True)
        else:
            self.runner.prefill(work)
        self.scheduler.record_work_duration(work, time.perf_counter() - start)
        self._last_work_kind = work.kind
        return (EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind),)

    def _run_decode(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        start = time.perf_counter()
        decode_batch = getattr(self.runner, "decode_batch", None)
        if callable(decode_batch):
            generated = tuple(decode_batch(work, commit=True))
        else:
            generated = tuple(self.runner.decode(work))
        self.scheduler.record_work_duration(work, time.perf_counter() - start)
        generated_events = self.scheduler.record_generated_events(generated)
        self._last_work_kind = work.kind
        events = [EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind)]
        for token_event in generated_events:
            events.append(
                EngineLoopEvent(
                    kind="token",
                    request_id=token_event.request_id,
                    request_ids=(token_event.request_id,),
                    token_id=token_event.token_id,
                    stream_chunk=token_event.stream_chunk,
                )
            )
        for item in (event.completed for event in generated_events if event.completed is not None):
            events.append(
                EngineLoopEvent(
                    kind="completed",
                    request_id=item.request_id,
                    request_ids=(item.request_id,),
                    completed=item,
                )
            )
        return tuple(events)

    def _reclaim_runner_state(self, completed: CompletedRequest) -> None:
        reclaim = getattr(self.runner, "reclaim", None)
        if callable(reclaim):
            reclaim(completed)


__all__ = [
    "DEFAULT_KV_POOL_CHUNK_PAGES",
    "DEFAULT_KV_POOL_IDLE_GRACE_SECONDS",
    "DEFAULT_KV_POOL_INITIAL_PAGES",
    "DEFAULT_KV_POOL_LOW_WATER_PAGES",
    "EngineLoopConfig",
    "EngineLoopEvent",
    "GenerationSubmission",
    "EngineLoopRunner",
    "PREFILL_DECODE_POLICIES",
    "ResidentEngineLoop",
    "SubmitPollTextGenerator",
    "add_engine_loop_config_args",
    "engine_loop_config_from_args",
    "engine_loop_config_from_env",
]
