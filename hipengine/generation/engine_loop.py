"""Long-lived scheduler-owned generation loop scaffolding.

This module is intentionally host-only and torch-free.  It wires the existing
``ResidentBatchScheduler`` to a small runner protocol so tests and early server
adapters can exercise a persistent ``submit``/``poll``/``cancel`` lifecycle
before native c>N sessions become correctness-green.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Protocol, Sequence

from hipengine.dispatch import RequestState, SlotMove, WorkItem, WorkKind
from hipengine.generation.batch_scheduler import (
    CompletedRequest,
    GeneratedToken,
    PerRowSamplingParams,
    ResidentBatchScheduler,
)
from hipengine.generation.deadline import GenerationCancelled, generation_deadline_expired
from hipengine.kvcache import (
    PREFIX_CACHE_CHOICES,
    ResourceUnavailable,
    resolve_prefix_cache_mode,
)
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    TextGenerator,
)
from hipengine.generation.sampling import speculative_mtp_sampling_blockers
from hipengine.speculative.frontier import (
    CandidateGraph,
    SpeculativeCapability,
    TargetFrontier,
)
from hipengine.speculative.policy import plan_speculative_requests
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    compose_speculative_claims,
)

PREFILL_DECODE_POLICIES = ("protect_decode", "protect_ttft", "fair", "token_budget")
DEFAULT_KV_POOL_INITIAL_PAGES = 128
DEFAULT_KV_POOL_LOW_WATER_PAGES = 128
DEFAULT_KV_POOL_CHUNK_PAGES = 128
DEFAULT_KV_POOL_IDLE_GRACE_SECONDS = 30.0
DEFAULT_MAX_PREFILL_CHUNK_TOKENS = 256
DEFAULT_ROUND_PREFILL_TOKEN_BUDGET = 1024
DEFAULT_ROUND_DECODE_ROW_BUDGET = 32
# Internal cross-thread routing absorbs transient scheduler bursts; the HTTP
# client-facing queue remains independently bounded by ServerConfig (default 16).
DEFAULT_RESIDENT_STREAM_QUEUE_MAX_CHUNKS = 64


@dataclass(frozen=True, slots=True)
class EngineLoopConfig:
    """CLI/env-resolved knobs for the C4 scheduler-owned engine loop."""

    prefill_decode_policy: str = "protect_decode"
    max_active_requests: int | None = None
    max_prefill_chunk_tokens: int = DEFAULT_MAX_PREFILL_CHUNK_TOKENS
    fair_prefill_burst_chunks: int = 1
    round_prefill_token_budget: int = DEFAULT_ROUND_PREFILL_TOKEN_BUDGET
    round_decode_row_budget: int = DEFAULT_ROUND_DECODE_ROW_BUDGET
    kv_pool_initial_pages: int = DEFAULT_KV_POOL_INITIAL_PAGES
    kv_pool_low_water_pages: int = DEFAULT_KV_POOL_LOW_WATER_PAGES
    kv_pool_high_water_pages: int | None = None
    kv_pool_chunk_pages: int = DEFAULT_KV_POOL_CHUNK_PAGES
    kv_pool_idle_grace_seconds: float = DEFAULT_KV_POOL_IDLE_GRACE_SECONDS
    max_pending_requests: int | None = None
    prefix_cache: str = "off"

    def __post_init__(self) -> None:
        if self.prefill_decode_policy not in PREFILL_DECODE_POLICIES:
            raise ValueError(f"prefill_decode_policy must be one of {PREFILL_DECODE_POLICIES!r}")
        if self.max_active_requests is not None and self.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive when set")
        if self.max_prefill_chunk_tokens <= 0:
            raise ValueError("max_prefill_chunk_tokens must be positive")
        if self.fair_prefill_burst_chunks <= 0:
            raise ValueError("fair_prefill_burst_chunks must be positive")
        if self.round_prefill_token_budget <= 0:
            raise ValueError("round_prefill_token_budget must be positive")
        if self.round_decode_row_budget <= 0:
            raise ValueError("round_decode_row_budget must be positive")
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
        object.__setattr__(self, "prefix_cache", resolve_prefix_cache_mode(self.prefix_cache))


class GenerationAdmissionRejected(MemoryError):
    """Retryable failure to reserve bounded generation resources at admission."""

    def __init__(
        self,
        message: str,
        *,
        resource: str,
        request_id: int | None = None,
        requested_units: int | None = None,
        current_units: int | None = None,
        capacity_units: int | None = None,
    ) -> None:
        if not str(message):
            raise ValueError("admission rejection message must not be empty")
        if not str(resource):
            raise ValueError("admission rejection resource must not be empty")
        for label, value in (
            ("requested_units", requested_units),
            ("current_units", current_units),
            ("capacity_units", capacity_units),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"{label} must be non-negative when set")
        self.resource = str(resource)
        self.request_id = None if request_id is None else int(request_id)
        if self.request_id is not None and self.request_id < 0:
            raise ValueError("request_id must be non-negative when set")
        self.requested_units = None if requested_units is None else int(requested_units)
        self.current_units = None if current_units is None else int(current_units)
        self.capacity_units = None if capacity_units is None else int(capacity_units)
        super().__init__(str(message))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "requested_units": self.requested_units,
            "current_units": self.current_units,
            "capacity_units": self.capacity_units,
        }


@dataclass(frozen=True, slots=True)
class GenerationSubmission:
    """Stable request ids for one batch submitted to a shared model loop."""

    request_ids: tuple[int, ...]
    request: GenerationRequest
    max_ticks: int
    work_kind: str = "decode"
    execution_route: str = "resident_scheduler"
    prelaunch_fallback: str | None = None
    work_item: WorkItem | None = None


@dataclass(slots=True)
class _ResidentStreamState:
    submission: GenerationSubmission
    events: deque[tuple[int, GenerationStreamChunk]] = field(default_factory=deque)
    pending_stop_chunks_by_request: dict[
        int,
        list[tuple[int, GenerationStreamChunk]],
    ] = field(default_factory=dict)
    emitted_text_request_ids: set[int] = field(default_factory=set)
    emitted_terminal_request_ids: set[int] = field(default_factory=set)
    overflowed_request_ids: set[int] = field(default_factory=set)
    cancelled_details: FinishDetails | None = None


class _SubmissionPriority:
    """Let queued admissions acquire the model-loop lock before the next tick."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting = 0

    @property
    def waiting_count(self) -> int:
        with self._condition:
            return self._waiting

    @contextmanager
    def submission(self, loop_lock: threading.Lock) -> Iterator[None]:
        with self._condition:
            self._waiting += 1
        try:
            loop_lock.acquire()
        except BaseException:
            with self._condition:
                self._waiting -= 1
                self._condition.notify_all()
            raise
        with self._condition:
            self._waiting -= 1
            self._condition.notify_all()
        try:
            yield
        finally:
            loop_lock.release()

    def wait_for_submissions(self) -> None:
        with self._condition:
            while self._waiting:
                self._condition.wait()


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
    error: BaseException | None = None


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

    def speculative_capability(
        self,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability | None:
        """Return one cold composed capability or None for target-only AR."""

    def speculative_claims_fit(self, plan) -> bool:
        """Check complete target/provider/transient fit without mutation."""

    def execute_speculative_cycle(
        self,
        plan,
        *,
        commit: bool,
    ) -> SpecCycleResult:
        """Execute exactly one bounded planned cycle."""


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
        for name in (
            "execution_profile",
            "execution_profile_manifest",
            "execution_profile_manifest_sha256",
            "execution_profile_strict_manifest_sha256",
            "execution_profile_fell_back_to_strict",
        ):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))
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
        if has_resident_runner:
            # Global pools/session slabs are a model-load responsibility. Doing
            # this before EngineService starts keeps submit commands O(1) and
            # avoids applying the short command timeout to first-use allocation.
            prepare_runner = getattr(self._runner, "prepare", None)
            if callable(prepare_runner):
                prepare_runner()
        # The lock protects one mutable scheduler tick, never an entire request.
        # Native runners therefore release it after each model transition so a
        # later D2 admission worker can enqueue between decode steps.
        self._loop_lock = threading.Lock()
        self._submission_priority = _SubmissionPriority()
        self._stream_queue_max_chunks = int(stream_queue_max_chunks)
        self._stream_states_by_request: dict[int, _ResidentStreamState] = {}
        self._submissions_by_request: dict[int, GenerationSubmission] = {}
        self._speculative_outputs_by_request: dict[int, GenerationOutput] = {}
        self._next_speculative_request_id = 1 << 60
        self._last_speculative_submission: GenerationSubmission | None = None
        self._cancel_dispatch_by_submission: dict[
            int,
            tuple[Any, Callable[[FinishDetails], None]],
        ] = {}
        self._cancel_commands_lock = threading.Lock()
        self._cancel_commands: deque[tuple[GenerationSubmission, FinishDetails]] = deque()

    @property
    def inner(self) -> TextGenerator:
        return self._inner

    @property
    def resident_capacity(self) -> int:
        """Return the immutable physical owner capacity used for admission."""

        return int(self._runner.capacity)

    @property
    def canonical_token_events(self) -> bool:
        """Whether scheduler token events are real generated-token events."""

        return self._has_resident_runner

    @property
    def supports_speculative_mtp(self) -> bool:
        """Whether staged or legacy speculative MTP is available."""

        if self._supports_staged_speculative_mtp:
            return True
        supports = getattr(self._inner, "supports_speculative_mtp", None)
        return bool(supports) and callable(
            getattr(self._inner, "generate_speculative_mtp_detailed", None)
        )

    @property
    def _supports_staged_speculative_mtp(self) -> bool:
        if not self._has_resident_runner:
            return False
        capability = callable(getattr(self._runner, "speculative_capability", None))
        opaque = callable(getattr(self._runner, "execute_speculative_cycle", None))
        staged = callable(getattr(self._runner, "execute_target_frontier", None))
        return capability and (opaque or staged)

    @property
    def last_speculative_submission(self) -> GenerationSubmission | None:
        return self._last_speculative_submission

    def generate_speculative_mtp_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        """Legacy exact pre-launch fallback for drivers without submission support."""

        if not self.supports_speculative_mtp:
            raise NotImplementedError(
                "speculative MTP generation is not supported by the wrapped generator"
            )
        return list(self._inner.generate_speculative_mtp_detailed(request))

    def submit_speculative_many_detailed(
        self,
        requests: Sequence[GenerationRequest],
    ) -> tuple[GenerationSubmission, ...]:
        """Pack compatible phase-serial speculative children into one model call."""

        normalized = tuple(requests)
        if not normalized:
            return ()
        if not self.supports_speculative_mtp:
            raise NotImplementedError(
                "speculative MTP generation is not supported by the wrapped generator"
            )
        if any(len(request.prompts) != 1 for request in normalized):
            raise ValueError("one speculative child requires exactly one prompt")
        if self._supports_staged_speculative_mtp:
            with self._submission_priority.submission(self._loop_lock):
                staged_submissions: list[GenerationSubmission] = []
                try:
                    for request in normalized:
                        staged_submissions.append(
                            self._submit_staged_speculative_detailed_locked(request)
                        )
                except Exception:
                    for submission in staged_submissions:
                        self._abort_submission_locked(submission, reason="cancel")
                    raise
                submissions = tuple(staged_submissions)
                if submissions:
                    self._last_speculative_submission = submissions[-1]
                return submissions
        compatibility = tuple(
            replace(request, prompts=(), cancellation_token=None)
            for request in normalized
        )
        if any(item != compatibility[0] for item in compatibility[1:]):
            return tuple(self.submit_speculative_detailed(request) for request in normalized)
        with self._submission_priority.submission(self._loop_lock):
            combined = replace(
                normalized[0],
                prompts=tuple(request.prompts[0] for request in normalized),
                cancellation_token=None,
            )
            outputs = list(self._inner.generate_speculative_mtp_detailed(combined))
            if len(outputs) != len(normalized):
                raise RuntimeError(
                    "packed speculative generation must return one output per child"
                )
            request_ids = tuple(
                range(
                    self._next_speculative_request_id,
                    self._next_speculative_request_id + len(normalized),
                )
            )
            self._next_speculative_request_id += len(normalized)
            details = getattr(self._inner, "last_batch_generation", {})
            spec_details = (
                details.get("speculative_mtp", {}) if isinstance(details, Mapping) else {}
            )
            draft_depth = max(
                1,
                int(spec_details.get("draft_n_max", min(3, normalized[0].max_tokens))),
            )
            work_item = WorkItem(
                kind=WorkKind.VERIFY_CHAIN,
                request_ids=request_ids,
                row_to_request=tuple(
                    request_id
                    for request_id in request_ids
                    for _ in range(draft_depth)
                ),
                token_rows=tuple(() for _ in range(len(request_ids) * draft_depth)),
                draft_depth=draft_depth,
                tree_parents=tuple(
                    parent
                    for _request_id in request_ids
                    for parent in range(draft_depth)
                ),
            )
            submissions: list[GenerationSubmission] = []
            for request_id, request, output in zip(
                request_ids, normalized, outputs, strict=True
            ):
                submission = GenerationSubmission(
                    request_ids=(request_id,),
                    request=request,
                    max_ticks=1,
                    work_kind=WorkKind.VERIFY_CHAIN.value,
                    execution_route="engine_service_packed_speculative",
                    work_item=work_item,
                )
                self._speculative_outputs_by_request[request_id] = output
                self._register_submission_cancellation_locked(submission)
                submissions.append(submission)
            self._last_speculative_submission = submissions[-1]
            return tuple(submissions)

    def submit_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> GenerationSubmission:
        """Submit one phase-serial VERIFY_CHAIN child to the shared service table.

        SPEC-C1 keeps the validated model-owned target cycle intact, but gives it
        stable request ownership and the normal EngineService collector/output
        lifecycle. Cross-request proposal/verify packing remains SPEC-C2.
        """

        if not self.supports_speculative_mtp:
            raise NotImplementedError(
                "speculative MTP generation is not supported by the wrapped generator"
            )
        if len(request.prompts) != 1:
            raise ValueError("one speculative submission requires exactly one prompt")
        with self._submission_priority.submission(self._loop_lock):
            if self._supports_staged_speculative_mtp:
                submission = self._submit_staged_speculative_detailed_locked(request)
                self._last_speculative_submission = submission
                return submission
            outputs = list(self._inner.generate_speculative_mtp_detailed(request))
            if len(outputs) != 1:
                raise RuntimeError("one speculative submission must produce one output")
            request_id = self._next_speculative_request_id
            self._next_speculative_request_id += 1
            details = getattr(self._inner, "last_batch_generation", {})
            spec_details = (
                details.get("speculative_mtp", {})
                if isinstance(details, Mapping)
                else {}
            )
            draft_depth = max(
                1,
                int(spec_details.get("draft_n_max", min(3, request.max_tokens))),
            )
            work_item = WorkItem(
                kind=WorkKind.VERIFY_CHAIN,
                request_ids=(request_id,),
                row_to_request=(request_id,) * draft_depth,
                token_rows=tuple(() for _ in range(draft_depth)),
                draft_depth=draft_depth,
                tree_parents=tuple(range(draft_depth)),
            )
            submission = GenerationSubmission(
                request_ids=(request_id,),
                request=request,
                max_ticks=1,
                work_kind=WorkKind.VERIFY_CHAIN.value,
                execution_route="engine_service_speculative",
                work_item=work_item,
            )
            self._speculative_outputs_by_request[request_id] = outputs[0]
            self._last_speculative_submission = submission
            self._register_submission_cancellation_locked(submission)
            return submission

    def _submit_staged_speculative_detailed_locked(
        self,
        request: GenerationRequest,
    ) -> GenerationSubmission:
        """Admit one staged child without running provider or target work."""

        prompt = request.prompts[0]
        tokenize_started = time.perf_counter() if isinstance(prompt, str) else None
        prompt_row = tuple(self._runner.prompt_tokens(prompt))
        tokenize_ms = (
            max(0.0, float(getattr(prompt, "tokenize_ms", 0.0)))
            if tokenize_started is None
            else max(0.0, (time.perf_counter() - tokenize_started) * 1_000.0)
        )
        max_new_tokens = int(self._runner.scheduler_max_new_tokens(request))
        desired_depth = getattr(
            self._runner,
            "speculative_desired_candidate_count",
            None,
        )
        desired = (
            int(desired_depth(request))
            if callable(desired_depth)
            else min(3, max_new_tokens)
        )
        token = request.cancellation_token

        def cancel_requested() -> bool:
            return generation_deadline_expired(request.deadline_at) or bool(
                token is not None
                and (
                    getattr(token, "cancel_requested", False)
                    or getattr(token, "cancelled", False)
                )
            )

        request_id: int | None = None
        try:
            request_id = self._loop.submit_speculative(
                prompt_row,
                max_new_tokens=max_new_tokens,
                desired_candidate_count=max(1, desired),
                cancel_requested=cancel_requested,
                sampling=PerRowSamplingParams.from_generation_request(request),
            )
            runner_request = replace(request, deadline_at=None)
            self._runner.register_batch(
                (request_id,),
                runner_request,
                prompt_rows=(prompt_row,),
            )
            register_speculative = getattr(
                self._runner,
                "register_speculative_request",
                None,
            )
            if callable(register_speculative):
                register_speculative(
                    request_id,
                    max(1, desired),
                    static_eligibility=request.speculative_mtp_static_eligibility,
                )
            timing_observer = getattr(
                self._runner,
                "record_prompt_tokenize_ms",
                None,
            )
            if callable(timing_observer):
                timing_observer((request_id,), (tokenize_ms,))
        except Exception:
            if request_id is not None:
                self._loop.cancel(request_id)
                self._loop.release_completed(request_id)
                self._runner.discard((request_id,))
            raise
        assert request_id is not None
        submission = GenerationSubmission(
            request_ids=(request_id,),
            request=request,
            max_ticks=_submit_poll_max_ticks(
                (prompt_row,),
                self._prefill_chunk_size,
                max_new_tokens=max_new_tokens,
                prefill_decode_policy=self._loop.prefill_decode_policy,
            ),
            work_kind=WorkKind.VERIFY_CHAIN.value,
            execution_route="engine_service_specdec2",
            work_item=None,
        )
        self._register_submission_cancellation_locked(submission)
        return submission

    @staticmethod
    def _is_staged_speculative_submission(
        submission: GenerationSubmission,
    ) -> bool:
        return submission.execution_route == "engine_service_specdec2"

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def server_mtp_batch_max_active_requests(self) -> int | None:
        """Expose the resident runner's explicit-MTP batch-route width bound."""

        return getattr(
            self._runner,
            "server_mtp_batch_max_active_requests",
            None,
        )

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
                if _events_advance_submission_tick(events, submission.request_ids):
                    ticks += 1
                if not events:
                    if self.generation_complete(submission):
                        continue
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

        with self._submission_priority.submission(self._loop_lock):
            return self._submit_detailed_locked(request)

    def _submit_detailed_locked(self, request: GenerationRequest) -> GenerationSubmission:
        prompts = tuple(request.prompts)
        if not prompts:
            raise ValueError("submit_detailed requires at least one prompt")
        normalized = replace(request, prompts=prompts)
        prompt_rows_list: list[tuple[int, ...]] = []
        tokenize_ms: list[float] = []
        for prompt in prompts:
            tokenize_started = time.perf_counter() if isinstance(prompt, str) else None
            prompt_rows_list.append(tuple(self._runner.prompt_tokens(prompt)))
            tokenize_ms.append(
                max(0.0, float(getattr(prompt, "tokenize_ms", 0.0)))
                if tokenize_started is None
                else max(0.0, (time.perf_counter() - tokenize_started) * 1_000.0)
            )
        prompt_rows = tuple(prompt_rows_list)
        max_new_tokens = int(self._runner.scheduler_max_new_tokens(normalized))
        request_ids: list[int] = []
        try:
            for prompt_row in prompt_rows:
                request_ids.append(
                    self._loop.submit(prompt_row, max_new_tokens=max_new_tokens)
                )
            # Native resident scheduling owns row deadlines/cancellation. Keep
            # the original request on the submission, but prevent one row's
            # expired deadline from raising through a shared physical model
            # batch. The serial compatibility bridge keeps its prior direct
            # token/deadline contract with the inner generator.
            runner_request = (
                replace(normalized, deadline_at=None)
                if self._has_resident_runner
                else normalized
            )
            self._runner.register_batch(
                request_ids,
                runner_request,
                prompt_rows=prompt_rows,
            )
            timing_observer = getattr(
                self._runner,
                "record_prompt_tokenize_ms",
                None,
            )
            if callable(timing_observer):
                timing_observer(request_ids, tokenize_ms)
        except Exception:
            for request_id in request_ids:
                self._loop.cancel(request_id)
                self._loop.release_completed(request_id)
            self._runner.discard(request_ids)
            raise
        submission = GenerationSubmission(
            request_ids=tuple(request_ids),
            request=normalized,
            max_ticks=_submit_poll_max_ticks(
                prompt_rows,
                self._prefill_chunk_size,
                max_new_tokens=max_new_tokens,
                prefill_decode_policy=self._loop.prefill_decode_policy,
            ),
        )
        self._register_submission_cancellation_locked(submission)
        return submission

    def poll(self, *, max_ticks: int = 1) -> tuple[EngineLoopEvent, ...]:
        """Advance shared model work without owning a request-lifetime lock."""

        self._submission_priority.wait_for_submissions()
        with self._loop_lock:
            self._drain_cancel_commands_locked()
            try:
                events = self._loop.poll(max_ticks=max_ticks)
            except GenerationAdmissionRejected as exc:
                request_id = exc.request_id
                if request_id is None:
                    pending = tuple(self._loop.scheduler.pending_requests)
                    request_id = None if not pending else int(pending[0].request_id)
                submission = (
                    None
                    if request_id is None
                    else self._submissions_by_request.get(int(request_id))
                )
                if request_id is None or submission is None:
                    raise
                self._abort_submission_locked(submission, reason="cancel")
                events = (
                    EngineLoopEvent(
                        kind="rejected",
                        request_id=int(request_id),
                        request_ids=(int(request_id),),
                        error=exc,
                    ),
                )
            self._route_stream_events_locked(events)
            return events

    def generation_complete(self, submission: GenerationSubmission) -> bool:
        with self._loop_lock:
            if (
                submission.work_kind in {
                    WorkKind.VERIFY_CHAIN.value,
                    WorkKind.VERIFY_TREE.value,
                }
                and not self._is_staged_speculative_submission(submission)
            ):
                return all(
                    request_id in self._speculative_outputs_by_request
                    for request_id in submission.request_ids
                )
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

    def _register_submission_cancellation_locked(
        self,
        submission: GenerationSubmission,
    ) -> None:
        if not self._has_resident_runner:
            return
        for request_id in submission.request_ids:
            self._submissions_by_request[int(request_id)] = submission
        token = submission.request.cancellation_token
        set_dispatch = getattr(token, "set_cancel_dispatch", None)
        if not callable(set_dispatch):
            return

        def dispatch(details: FinishDetails) -> None:
            with self._cancel_commands_lock:
                self._cancel_commands.append(
                    (submission, FinishDetails.from_value(details))
                )

        set_dispatch(dispatch)
        self._cancel_dispatch_by_submission[id(submission)] = (token, dispatch)
        if bool(getattr(token, "cancelled", False)):
            dispatch(FinishDetails.from_value(getattr(token, "finish_details", None)))

    def _unregister_submission_cancellation_locked(
        self,
        submission: GenerationSubmission,
    ) -> None:
        for request_id in submission.request_ids:
            if self._submissions_by_request.get(int(request_id)) is submission:
                self._submissions_by_request.pop(int(request_id), None)
        registered = self._cancel_dispatch_by_submission.pop(id(submission), None)
        if registered is None:
            return
        token, dispatch = registered
        clear_dispatch = getattr(token, "clear_cancel_dispatch", None)
        if callable(clear_dispatch):
            clear_dispatch(dispatch)

    def drain_cancellations(self) -> int:
        """Acknowledge queued row cancellation when no stream remains to poll."""

        with self._loop_lock:
            with self._cancel_commands_lock:
                queued = len(self._cancel_commands)
            self._drain_cancel_commands_locked()
            return queued

    def _drain_cancel_commands_locked(self) -> None:
        with self._cancel_commands_lock:
            queued_commands = tuple(self._cancel_commands)
            self._cancel_commands.clear()
        commands_by_submission = {
            id(submission): (submission, details)
            for submission, details in queued_commands
        }
        active_submissions = {
            id(submission): submission
            for submission in self._submissions_by_request.values()
        }
        for submission_id, submission in active_submissions.items():
            if generation_deadline_expired(submission.request.deadline_at):
                commands_by_submission.setdefault(
                    submission_id,
                    (
                        submission,
                        FinishDetails(
                            reason="deadline_exceeded",
                            deadline_exceeded=True,
                        ),
                    ),
                )
        for submission, details in commands_by_submission.values():
            active = any(
                self._submissions_by_request.get(int(request_id)) is submission
                for request_id in submission.request_ids
            )
            if active:
                reason = "timeout" if details.deadline_exceeded else "disconnect"
                state = next(
                    (
                        self._stream_states_by_request.get(int(request_id))
                        for request_id in submission.request_ids
                        if self._stream_states_by_request.get(int(request_id)) is not None
                    ),
                    None,
                )
                if state is not None:
                    state.cancelled_details = details
                    state.events.clear()
                    self._unregister_stream_state_locked(state)
                    self._abort_submission_locked(submission, reason=reason)
                else:
                    for request_id in submission.request_ids:
                        self._loop.cancel(int(request_id), reason=reason)
                        self._loop.release_completed(int(request_id))
                    self._unregister_submission_cancellation_locked(submission)
            token = submission.request.cancellation_token
            acknowledge = getattr(token, "acknowledge_cancel", None)
            if callable(acknowledge):
                acknowledge(details)

    def take_result(self, submission: GenerationSubmission) -> list[GenerationOutput]:
        """Consume one completed submission in original prompt order."""

        with self._loop_lock:
            if (
                submission.work_kind in {
                    WorkKind.VERIFY_CHAIN.value,
                    WorkKind.VERIFY_TREE.value,
                }
                and not self._is_staged_speculative_submission(submission)
            ):
                if not all(
                    request_id in self._speculative_outputs_by_request
                    for request_id in submission.request_ids
                ):
                    raise RuntimeError("submitted speculative generation is incomplete")
                outputs = [
                    self._speculative_outputs_by_request.pop(request_id)
                    for request_id in submission.request_ids
                ]
                self._unregister_submission_cancellation_locked(submission)
                return outputs
            if not self._runner.has_outputs(submission.request_ids):
                missing = self._runner.missing_outputs(submission.request_ids)
                raise RuntimeError(f"submitted generation is incomplete; missing request_ids={missing}")
            outputs = self._runner.take_outputs(submission.request_ids)
            for request_id in submission.request_ids:
                self._loop.release_completed(request_id)
            finalize_batch = getattr(self._runner, "finalize_batch", None)
            if callable(finalize_batch):
                finalize_batch(submission.request, submission.request_ids, outputs)
            self._unregister_submission_cancellation_locked(submission)
            return outputs

    def abort_submission(
        self,
        submission: GenerationSubmission,
        *,
        reason: str = "cancel",
    ) -> None:
        """Synchronously reclaim one child submission on the sole driver thread."""

        with self._loop_lock:
            self._abort_submission_locked(submission, reason=str(reason))

    def _abort_submission(self, submission: GenerationSubmission) -> None:
        self.abort_submission(submission, reason="cancel")

    def reconfigure_engine_loop(self, config: EngineLoopConfig) -> None:
        """Apply one idle configuration on the sole scheduler driver."""

        with self._loop_lock:
            self._loop.reconfigure(config)
            self._prefill_chunk_size = int(self._loop.prefill_chunk_size)

    def compact(self, order: Sequence[int] | None = None) -> tuple[SlotMove, ...]:
        """Serialize scheduler/model compaction with submit/poll ownership."""

        requested = None if order is None else tuple(int(value) for value in order)
        with self._loop_lock:
            return tuple(self._loop.compact(order=requested))

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
                generated_token_ids=generation_output.generated_token_ids,
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

        with self._submission_priority.submission(self._loop_lock):
            submission = self._submit_detailed_locked(request)
            state = _ResidentStreamState(submission)
            for request_id in submission.request_ids:
                if request_id in self._stream_states_by_request:
                    self._abort_submission_locked(submission, reason="cancel")
                    raise RuntimeError(f"request_id {request_id} already has a stream subscription")
                self._stream_states_by_request[request_id] = state

        consumed = False
        cancel_acknowledged = False
        ticks = 0
        try:
            while True:
                queued: tuple[int, GenerationStreamChunk] | None = None
                cancelled_details: FinishDetails | None = None
                with self._loop_lock:
                    token = submission.request.cancellation_token
                    if state.cancelled_details is not None:
                        cancelled_details = state.cancelled_details
                        cancel_acknowledged = True
                        complete = True
                    elif token is not None and bool(getattr(token, "cancelled", False)):
                        # Fallback for cancellation tokens without scheduler
                        # dispatch support. Native server tokens use the queued
                        # command/ack path drained by ``poll()``.
                        cancelled_details = FinishDetails.from_value(
                            getattr(token, "finish_details", None)
                        )
                        reason = (
                            "timeout"
                            if cancelled_details.deadline_exceeded
                            else "disconnect"
                        )
                        self._abort_submission_locked(submission, reason=reason)
                        self._unregister_stream_state_locked(state)
                        state.events.clear()
                        cancel_acknowledged = True
                        complete = True
                    else:
                        if state.events:
                            queued = state.events.popleft()
                        complete = self._runner.has_outputs(submission.request_ids)
                if cancelled_details is not None:
                    raise GenerationCancelled(cancelled_details)
                if queued is not None:
                    request_id, chunk = queued
                    if chunk.text:
                        state.emitted_text_request_ids.add(request_id)
                    if chunk.finish_details is not None:
                        state.emitted_terminal_request_ids.add(request_id)
                    if chunk.text or chunk.token_logprobs or chunk.finish_details is not None:
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
                if _events_advance_submission_tick(events, submission.request_ids):
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
                if (
                    request_id in state.emitted_text_request_ids
                    or request_id in state.emitted_terminal_request_ids
                ):
                    continue
                yield GenerationStreamChunk(
                    text=generation_output.text,
                    token_logprobs=generation_output.token_logprobs,
                    finish_details=generation_output.finish_details,
                    telemetry=generation_output.telemetry,
                    generated_token_ids=generation_output.generated_token_ids,
                )
            consumed = True
        finally:
            if not consumed and not cancel_acknowledged:
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
            for stream_chunk in _stop_safe_resident_stream_chunks(state, event):
                if len(state.events) >= self._stream_queue_max_chunks:
                    state.overflowed_request_ids.add(request_id)
                    self._loop.cancel(request_id, reason="cancel")
                    break
                state.events.append((request_id, stream_chunk))

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
        if (
            submission.work_kind in {
                WorkKind.VERIFY_CHAIN.value,
                WorkKind.VERIFY_TREE.value,
            }
            and not self._is_staged_speculative_submission(submission)
        ):
            for request_id in submission.request_ids:
                self._speculative_outputs_by_request.pop(request_id, None)
            self._unregister_submission_cancellation_locked(submission)
            return
        for request_id in submission.request_ids:
            self._loop.cancel(request_id, reason=reason)
            self._loop.release_completed(request_id)
        self._runner.discard(submission.request_ids)
        self._unregister_submission_cancellation_locked(submission)

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
            submissions = {
                id(submission): submission
                for submission in self._submissions_by_request.values()
            }
            for submission in submissions.values():
                self._unregister_submission_cancellation_locked(submission)
            self._speculative_outputs_by_request.clear()
            with self._cancel_commands_lock:
                self._cancel_commands.clear()
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

    def close(self) -> None:
        """Release resources owned by the compatibility generator."""

        self._rows.clear()
        self._outputs.clear()
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()

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


def _stop_safe_resident_stream_chunks(
    state: _ResidentStreamState,
    event: EngineLoopEvent,
) -> tuple[GenerationStreamChunk, ...]:
    """Hold only token chunks that can still complete a configured stop."""

    assert event.request_id is not None and event.token_id is not None
    assert event.stream_chunk is not None
    request_id = int(event.request_id)
    chunk = event.stream_chunk
    pending = state.pending_stop_chunks_by_request.setdefault(request_id, [])
    pending.append((int(event.token_id), chunk))
    finish = chunk.finish_details
    if finish is not None:
        suppressed = _resident_stream_suppressed_suffix(
            pending,
            finish,
            state.submission.request,
        )
        if suppressed <= 0:
            output = tuple(item[1] for item in pending)
        else:
            output = (
                *(item[1] for item in pending[:-suppressed]),
                replace(chunk, text="", token_logprobs=()),
            )
        state.pending_stop_chunks_by_request.pop(request_id, None)
        return output

    prefixes = _resident_stream_proper_stop_prefixes(
        state.submission.request.stop_token_sequences
    )
    token_ids = tuple(item[0] for item in pending)
    retained = max(
        (
            len(prefix)
            for prefix in prefixes
            if len(prefix) <= len(token_ids)
            and token_ids[-len(prefix) :] == prefix
        ),
        default=0,
    )
    emit_count = len(pending) - retained
    output = tuple(item[1] for item in pending[:emit_count])
    del pending[:emit_count]
    return output


def _resident_stream_proper_stop_prefixes(
    stop_sequences: Sequence[Sequence[int]],
) -> frozenset[tuple[int, ...]]:
    return frozenset(
        tuple(int(token) for token in sequence[:width])
        for sequence in stop_sequences
        for width in range(1, len(sequence))
    )


def _resident_stream_suppressed_suffix(
    pending: Sequence[tuple[int, GenerationStreamChunk]],
    finish: FinishDetails,
    request: GenerationRequest,
) -> int:
    if finish.stop_sequence:
        return min(len(pending), len(finish.stop_sequence))
    token_ids = tuple(item[0] for item in pending)
    if finish.reason == "stop":
        for sequence in request.stop_token_sequences:
            normalized = tuple(int(token) for token in sequence)
            if normalized and len(normalized) <= len(token_ids) and token_ids[
                -len(normalized) :
            ] == normalized:
                return len(normalized)
        if token_ids and token_ids[-1] in set(request.stop_token_ids):
            return 1
    if finish.reason == "eos" and finish.eos_token_id is not None:
        return 1
    return 0


def _events_advance_submission_tick(
    events: Sequence[EngineLoopEvent],
    request_ids: Sequence[int],
) -> bool:
    """Return whether one shared-loop poll advanced this submission's work.

    Concurrent streams may drive scheduler ticks that contain only a longer or
    otherwise preferred peer. Those peer-only ticks cannot consume the local
    finite-work budget derived from this submission's own prompt and decode
    lengths.
    """

    owned = frozenset(int(request_id) for request_id in request_ids)
    return bool(owned) and any(
        not owned.isdisjoint(int(request_id) for request_id in event.request_ids)
        for event in events
    )


def _submit_poll_max_ticks(
    prompt_rows: Sequence[Sequence[int]],
    prefill_chunk_size: int,
    *,
    max_new_tokens: int,
    prefill_decode_policy: str = "fair",
) -> int:
    """Return a finite bound that covers real chunked prefill and token decode."""

    chunk_size = int(prefill_chunk_size)
    if chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    prefill_ticks = sum(
        max(1, (len(row) + chunk_size - 1) // chunk_size)
        for row in prompt_rows
    )
    # Decode normally advances every ready row once per tick, so the longest
    # request—not the sum across rows—sets the steady decode bound. With
    # protect_decode, however, the first row can become ready before later
    # prompt chunks and then run to completion; the same submission can
    # therefore consume one full decode span per staggered row. This is only a
    # finite stall guard, not a scheduling target, so cover that exact worst
    # case rather than abort valid work near completion.
    policy = str(prefill_decode_policy)
    if policy not in PREFILL_DECODE_POLICIES:
        raise ValueError(f"unknown prefill/decode policy: {policy!r}")
    decode_ticks = max(1, int(max_new_tokens)) * (
        len(prompt_rows) if policy == "protect_decode" else 1
    )
    stagger_margin = max(len(prompt_rows) + 4, prefill_ticks)
    return max(8, prefill_ticks + decode_ticks + stagger_margin)


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
        "--fair-prefill-burst-chunks",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_FAIR_PREFILL_BURST_CHUNKS", 1),
        help="Maximum consecutive prefill chunks while fair scheduling has decode work (env HIPENGINE_FAIR_PREFILL_BURST_CHUNKS; default: 1)",
    )
    parser.add_argument(
        "--round-prefill-token-budget",
        type=_positive_int_arg,
        default=_env_positive_int(
            env,
            "HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET",
            DEFAULT_ROUND_PREFILL_TOKEN_BUDGET,
        ),
        help="Token-budget prefill work per round (env HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET; default: 1024)",
    )
    parser.add_argument(
        "--round-decode-row-budget",
        type=_positive_int_arg,
        default=_env_positive_int(
            env,
            "HIPENGINE_ROUND_DECODE_ROW_BUDGET",
            DEFAULT_ROUND_DECODE_ROW_BUDGET,
        ),
        help="Token-budget due decode rows per round (env HIPENGINE_ROUND_DECODE_ROW_BUDGET; default: 32)",
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
    parser.add_argument(
        "--prefix-cache",
        choices=PREFIX_CACHE_CHOICES,
        default=resolve_prefix_cache_mode(env.get("HIPENGINE_PREFIX_CACHE")),
        help="Prefix-cache mode (env HIPENGINE_PREFIX_CACHE; default: off)",
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
        fair_prefill_burst_chunks=int(getattr(args, "fair_prefill_burst_chunks")),
        round_prefill_token_budget=int(
            getattr(
                args,
                "round_prefill_token_budget",
                DEFAULT_ROUND_PREFILL_TOKEN_BUDGET,
            )
        ),
        round_decode_row_budget=int(
            getattr(
                args,
                "round_decode_row_budget",
                DEFAULT_ROUND_DECODE_ROW_BUDGET,
            )
        ),
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
        prefix_cache=resolve_prefix_cache_mode(getattr(args, "prefix_cache", "off")),
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

    Requests stay resident across polls and completion reclaim is delegated to
    ``ResidentBatchScheduler``. Legacy policies execute one work item per tick;
    ``token_budget`` executes fair prefill quanta plus one decode step for every
    due resident row in a bounded scheduling round.
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
        self.fair_prefill_burst_chunks = int(resolved_config.fair_prefill_burst_chunks)
        self.round_prefill_token_budget = int(resolved_config.round_prefill_token_budget)
        self.round_decode_row_budget = int(resolved_config.round_decode_row_budget)
        if (
            self.prefill_decode_policy == "token_budget"
            and self.round_decode_row_budget < int(resolved_capacity)
        ):
            raise ValueError(
                "token_budget round_decode_row_budget must cover resident capacity"
            )
        self._last_work_kind: WorkKind | None = None
        self._consecutive_prefill_chunks = 0
        self._cold_prefill_cohort_request_ids: frozenset[int] = frozenset()
        self._rounds = 0
        self._round_prefill_tokens = 0
        self._round_decode_rows = 0
        self.scheduler = ResidentBatchScheduler(
            capacity=resolved_capacity,
            context_bucket_size=context_bucket_size,
            max_pending_requests=resolved_config.max_pending_requests,
            reclaim_callback=self._reclaim_runner_state,
        )
        self._speculative_candidate_counts: dict[int, int] = {}
        self._speculative_cancel_probes: dict[int, Callable[[], bool]] = {}
        self._speculative_cycle_sequence = 0
        self._last_speculative_plan = None
        self._recent_speculative_plans = deque(maxlen=32)

    def reconfigure(self, config: EngineLoopConfig) -> None:
        """Apply an idle resource/policy generation without replacing the loop."""

        if self.scheduler.active_count or self.scheduler.pending_count:
            raise RuntimeError("cannot reconfigure a non-idle resident engine loop")
        capacity = int(self.scheduler.capacity)
        if (
            config.max_active_requests is not None
            and int(config.max_active_requests) != capacity
        ):
            raise ValueError("reconfiguration cannot change resident capacity")
        if (
            config.prefill_decode_policy == "token_budget"
            and int(config.round_decode_row_budget) < capacity
        ):
            raise ValueError(
                "token_budget round_decode_row_budget must cover resident capacity"
            )
        resource_fields = (
            "kv_pool_initial_pages",
            "kv_pool_low_water_pages",
            "kv_pool_high_water_pages",
            "kv_pool_chunk_pages",
            "kv_pool_idle_grace_seconds",
            "prefix_cache",
        )
        resource_changed = any(
            getattr(self.config, field_name) != getattr(config, field_name)
            for field_name in resource_fields
        )
        configure_runner = getattr(self.runner, "configure_engine_loop", None)
        if resource_changed and callable(configure_runner):
            configure_runner(config)
        self.config = config
        self.prefill_chunk_size = int(config.max_prefill_chunk_tokens)
        self.prefill_decode_policy = config.prefill_decode_policy
        self.fair_prefill_burst_chunks = int(config.fair_prefill_burst_chunks)
        self.round_prefill_token_budget = int(config.round_prefill_token_budget)
        self.round_decode_row_budget = int(config.round_decode_row_budget)
        self._last_work_kind = None
        self._consecutive_prefill_chunks = 0
        self._cold_prefill_cohort_request_ids = frozenset()
        self._round_prefill_tokens = 0
        self._round_decode_rows = 0
        self._last_speculative_plan = None
        self._recent_speculative_plans.clear()

    @property
    def last_speculative_plan(self):
        return self._last_speculative_plan

    @property
    def recent_speculative_plans(self):
        return tuple(self._recent_speculative_plans)

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
        resource_snapshot = getattr(self.runner, "resource_observability_snapshot", None)
        if callable(resource_snapshot):
            snapshot["resources"] = resource_snapshot()
        plan = self._last_speculative_plan
        snapshot["recent_speculative_plans"] = [
            {
                "request_ids": list(recent.request_ids),
                "candidate_counts": list(recent.candidate_counts),
                "reasons": [reason.value for reason in recent.reasons],
                "k0_classes": [value.value for value in recent.k0_classes],
                "execution_route": recent.execution_route,
            }
            for recent in self._recent_speculative_plans
        ]
        snapshot["last_speculative_plan"] = (
            None
            if plan is None
            else {
                "request_ids": list(plan.request_ids),
                "candidate_counts": list(plan.candidate_counts),
                "reasons": [reason.value for reason in plan.reasons],
                "k0_classes": [value.value for value in plan.k0_classes],
                "execution_route": plan.execution_route,
                "logical_frontier_rows": plan.logical_frontier_rows,
            }
        )
        snapshot["scheduler_policy"] = {
            "prefill_decode_policy": self.prefill_decode_policy,
            "prefill_chunk_tokens": int(self.prefill_chunk_size),
            "fair_prefill_burst_chunks": int(self.fair_prefill_burst_chunks),
            "round_prefill_token_budget": int(self.round_prefill_token_budget),
            "round_decode_row_budget": int(self.round_decode_row_budget),
            "rounds": int(self._rounds),
            "round_prefill_tokens": int(self._round_prefill_tokens),
            "round_decode_rows": int(self._round_decode_rows),
            "consecutive_prefill_chunks": int(self._consecutive_prefill_chunks),
            "cold_prefill_cohort_size": len(self._cold_prefill_cohort_request_ids),
            "last_work_kind": (
                None if self._last_work_kind is None else self._last_work_kind.value
            ),
        }
        return snapshot

    def submit(
        self,
        prompt_tokens: Iterable[int],
        *,
        max_new_tokens: int,
        request_id: int | None = None,
        sampling: PerRowSamplingParams | None = None,
    ) -> int:
        max_pending = self.scheduler.max_pending_requests
        pending = self.scheduler.pending_count
        if max_pending is not None and pending >= max_pending:
            raise GenerationAdmissionRejected(
                f"pending request queue is full (max_pending_requests={max_pending})",
                resource="pending_request_queue",
                requested_units=1,
                current_units=pending,
                capacity_units=max_pending,
            )
        return self.scheduler.submit(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            request_id=request_id,
            sampling=sampling,
        )

    def submit_speculative(
        self,
        prompt_tokens: Iterable[int],
        *,
        max_new_tokens: int,
        desired_candidate_count: int,
        request_id: int | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        sampling: PerRowSamplingParams | None = None,
    ) -> int:
        """Submit speculative intent into the ordinary Generation-2 lifecycle."""

        desired = int(desired_candidate_count)
        if desired <= 0:
            raise ValueError("desired_candidate_count must be positive")
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable")
        selected = self.submit(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            request_id=request_id,
            sampling=sampling,
        )
        self._speculative_candidate_counts[selected] = desired
        if cancel_requested is not None:
            self._speculative_cancel_probes[selected] = cancel_requested
        return selected

    def cancel(self, request_id: int, *, reason: str = "cancel") -> bool:
        """Cancel a pending or active request and reclaim active scheduler state."""

        completed = self.scheduler.cancel(request_id, reason=reason)
        if completed is not None:
            self._speculative_candidate_counts.pop(int(request_id), None)
            self._speculative_cancel_probes.pop(int(request_id), None)
        return completed is not None

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
        plan_admission = getattr(self.runner, "plan_admission", None)
        selected_request_ids: Sequence[int] | None = None
        free_slots = self.scheduler.capacity - self.scheduler.active_count
        if callable(plan_admission) and free_slots > 0 and self.scheduler.pending_count:
            try:
                selected_request_ids = tuple(
                    int(request_id)
                    for request_id in plan_admission(
                        self.scheduler.pending_requests,
                        max_items=free_slots,
                    )
                )
            except ResourceUnavailable as exc:
                raise GenerationAdmissionRejected(
                    str(exc),
                    resource=exc.resource,
                    requested_units=exc.requested_units,
                    current_units=exc.current_units,
                    capacity_units=exc.capacity_units,
                ) from exc
        admitted = self.scheduler.admit_pending(
            request_ids=selected_request_ids,
            reserve_callback=(reserve_admission if callable(reserve_admission) else None),
            rollback_callback=(rollback_admission if callable(rollback_admission) else None),
        )
        events.extend(
            EngineLoopEvent(kind="admitted", request_id=request_id, request_ids=(request_id,))
            for request_id in admitted
        )

        if self.prefill_decode_policy == "token_budget":
            events.extend(self._run_token_budget_round())
            return tuple(events)

        decode = self.scheduler.next_decode_work()
        prefill_available = self.scheduler.has_prefill_work()
        self._update_cold_prefill_cohort(
            decode_available=decode is not None,
            prefill_available=prefill_available,
        )
        if self._choose_decode_first(decode_available=decode is not None, prefill_available=prefill_available):
            assert decode is not None
            events.extend(self._run_decode(decode))
            return tuple(events)

        if prefill_available:
            packed_prefill_rows = int(
                getattr(self.runner, "packed_prefill_max_rows", 1)
            )
            if packed_prefill_rows > 1:
                prefill = self.scheduler.next_prefill_batch_work(
                    chunk_size=self.prefill_chunk_size,
                    max_rows=packed_prefill_rows,
                )
            else:
                prefill = self.scheduler.next_prefill_work(
                    chunk_size=self.prefill_chunk_size
                )
            assert prefill is not None
            events.extend(self._run_prefill(prefill))
            return tuple(events)

        if decode is None:
            return tuple(events)
        events.extend(self._run_decode(decode))
        return tuple(events)

    def _run_token_budget_round(self) -> tuple[EngineLoopEvent, ...]:
        """Run fair prefill quanta and one decode step for every due row."""

        events: list[EngineLoopEvent] = []
        prefill_budget = self.round_prefill_token_budget
        prefill_ran = False
        multiple_prefills = bool(
            getattr(self.runner, "supports_multiple_prefill_quanta_per_round", False)
        )
        while prefill_budget > 0 and self.scheduler.has_prefill_work():
            work = self.scheduler.next_round_robin_prefill_work(
                chunk_size=min(self.prefill_chunk_size, prefill_budget)
            )
            if work is None:
                break
            tokens = sum(len(row) for row in work.token_rows)
            if tokens <= 0 or tokens > prefill_budget:
                raise AssertionError("token-budget prefill planner exceeded its budget")
            events.extend(self._run_prefill(work))
            prefill_ran = True
            prefill_budget -= tokens
            self._round_prefill_tokens += tokens
            if not multiple_prefills:
                break

        same_round_decode = bool(
            getattr(self.runner, "supports_prefill_decode_same_round", False)
        )
        decode = (
            None
            if prefill_ran and not same_round_decode
            else self.scheduler.next_decode_work()
        )
        if decode is not None:
            if len(decode.request_ids) > self.round_decode_row_budget:
                raise AssertionError("due decode rows exceed the configured round budget")
            events.extend(self._run_decode(decode))
            self._round_decode_rows += len(decode.request_ids)
        self._rounds += 1
        return tuple(events)

    def _update_cold_prefill_cohort(
        self,
        *,
        decode_available: bool,
        prefill_available: bool,
    ) -> None:
        if self.prefill_decode_policy != "fair" or self.fair_prefill_burst_chunks <= 1:
            self._cold_prefill_cohort_request_ids = frozenset()
            return
        current = frozenset(self.scheduler.prefill_request_ids())
        if self._cold_prefill_cohort_request_ids:
            additions = current.difference(self._cold_prefill_cohort_request_ids)
            if additions and self.scheduler.prefill_requests_fit_within_chunks(
                chunk_size=self.prefill_chunk_size,
                max_chunks=self.fair_prefill_burst_chunks,
            ):
                # No member can free a physical slot before the first decode,
                # so bounded pre-decode additions cannot extend this epoch past
                # resident capacity. A decode clears the cohort below.
                self._cold_prefill_cohort_request_ids |= additions
            self._cold_prefill_cohort_request_ids &= current
            return
        if (
            not decode_available
            and prefill_available
            and self.scheduler.prefill_requests_fit_within_chunks(
                chunk_size=self.prefill_chunk_size,
                max_chunks=self.fair_prefill_burst_chunks,
            )
        ):
            self._cold_prefill_cohort_request_ids = current

    def _choose_decode_first(self, *, decode_available: bool, prefill_available: bool) -> bool:
        if not decode_available:
            return False
        if not prefill_available:
            return True
        if self.prefill_decode_policy == "protect_decode":
            return True
        if self.prefill_decode_policy == "protect_ttft":
            return False
        if self._cold_prefill_cohort_request_ids:
            return False
        # A multi-chunk interruption only earns its ITL cost when more than one
        # prompt remains: completing the current row then forms a wider group
        # while another prompt can use the next burst. Lone staggered arrivals
        # retain strict one-chunk fair alternation.
        burst_limit = (
            self.fair_prefill_burst_chunks
            if self.scheduler.prefill_request_count() > 1
            else 1
        )
        return self._consecutive_prefill_chunks >= burst_limit

    def _run_prefill(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        start = time.perf_counter()
        prefill_batch = getattr(self.runner, "prefill_batch", None)
        if callable(prefill_batch):
            prefill_batch(work, commit=True)
        else:
            self.runner.prefill(work)
        self.scheduler.record_work_duration(work, time.perf_counter() - start)
        self._last_work_kind = work.kind
        self._consecutive_prefill_chunks += 1
        return (EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind),)

    def _run_decode(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        desired = tuple(
            self._speculative_candidate_counts.get(int(request_id), 0)
            for request_id in work.request_ids
        )
        partitioned = self._maybe_run_partitioned_speculative_decode(
            work,
            desired,
        )
        if partitioned is not None:
            return partitioned
        speculative = self._maybe_run_speculative_cycle(work)
        if speculative is not None:
            return speculative
        if any(desired) and not all(desired):
            spec_ids = tuple(
                request_id
                for request_id, count in zip(work.request_ids, desired, strict=True)
                if count > 0
            )
            spec_work = self._decode_work_subset(work, spec_ids)
            spec_desired = tuple(
                self._speculative_candidate_counts[int(request_id)]
                for request_id in spec_ids
            )
            disjoint_speculative = self._maybe_run_partitioned_speculative_decode(
                spec_work,
                spec_desired,
            )
            if disjoint_speculative is None:
                disjoint_speculative = self._maybe_run_speculative_cycle(spec_work)
            if disjoint_speculative is not None:
                ar_ids = tuple(
                    request_id
                    for request_id, count in zip(
                        work.request_ids,
                        desired,
                        strict=True,
                    )
                    if count == 0
                )
                ar_events = self._run_ar_decode(
                    self._decode_work_subset(work, ar_ids)
                )
                return (*disjoint_speculative, *ar_events)
        return self._run_ar_decode(work)

    def _maybe_run_partitioned_speculative_decode(
        self,
        work: WorkItem,
        desired: Sequence[int],
    ) -> tuple[EngineLoopEvent, ...] | None:
        """Lower one wide all-spec due item into bounded fair frontiers.

        The resident scheduler remains the sole fairness owner: request order is
        stable, every due row is served exactly once in this tick, and each
        physical subgroup resolves its own immutable plan before mutation.
        """

        counts = tuple(int(value) for value in desired)
        if len(work.request_ids) <= 1 or not counts or not all(counts):
            return None
        resolve_width = getattr(
            self.runner,
            "speculative_partition_max_requests",
            None,
        )
        if not callable(resolve_width):
            return None
        max_requests = int(resolve_width(work))
        if max_requests <= 0 or len(work.request_ids) <= max_requests:
            return None
        events: list[EngineLoopEvent] = []
        for start in range(0, len(work.request_ids), max_requests):
            request_ids = work.request_ids[start : start + max_requests]
            subgroup = self._decode_work_subset(work, request_ids)
            speculative = self._maybe_run_speculative_cycle(subgroup)
            if speculative is None:
                events.extend(self._run_ar_decode(subgroup))
            else:
                events.extend(speculative)
        return tuple(events)

    @staticmethod
    def _decode_work_subset(
        work: WorkItem,
        request_ids: Sequence[int],
    ) -> WorkItem:
        selected = tuple(int(request_id) for request_id in request_ids)
        if not selected or any(request_id not in work.request_ids for request_id in selected):
            raise ValueError("decode work subset must be a non-empty request subset")
        selected_set = set(selected)
        row_indices = tuple(
            index
            for index, request_id in enumerate(work.row_to_request)
            if int(request_id) in selected_set
        )
        slots = ()
        if work.slot_ids:
            slot_by_request = dict(
                zip(work.request_ids, work.slot_ids, strict=True)
            )
            slots = tuple(int(slot_by_request[request_id]) for request_id in selected)
        selected_slots = set(slots)
        active_mask = (
            tuple(index in selected_slots for index in range(len(work.active_mask)))
            if work.active_mask
            else ()
        )
        return WorkItem(
            kind=work.kind,
            request_ids=selected,
            row_to_request=tuple(work.row_to_request[index] for index in row_indices),
            token_rows=tuple(work.token_rows[index] for index in row_indices)
            if work.token_rows
            else (),
            slot_ids=slots,
            active_mask=active_mask,
        )

    def _run_ar_decode(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        start = time.perf_counter()
        decode_batch = getattr(self.runner, "decode_batch", None)
        if callable(decode_batch):
            generated = tuple(decode_batch(work, commit=True))
        else:
            generated = tuple(self.runner.decode(work))
        self.scheduler.record_work_duration(work, time.perf_counter() - start)
        generated_events = self.scheduler.record_generated_events(generated)
        return self._decode_events(work, generated_events)

    def _maybe_run_speculative_cycle(
        self,
        work: WorkItem,
    ) -> tuple[EngineLoopEvent, ...] | None:
        desired = tuple(
            self._speculative_candidate_counts.get(int(request_id), 0)
            for request_id in work.request_ids
        )
        if not any(desired):
            return None
        suppression: tuple[bool, ...] = ()
        cooldown_probe = getattr(
            self.runner, "speculative_post_reject_cooldown", None
        )
        if callable(cooldown_probe):
            suppression = tuple(
                bool(flag)
                for flag in cooldown_probe(tuple(work.request_ids))
            )
            if len(suppression) != len(desired) or not any(suppression):
                suppression = ()
        semantics: list[SpeculativeRequestSemantics] = []
        sampler_block = self.scheduler.sampler_params_block(work.request_ids)
        for request_id in work.request_ids:
            request = self.scheduler.active_batch.requests[int(request_id)]
            params = sampler_block.params_for(int(request_id))
            sampling_mode = (
                "greedy"
                if not speculative_mtp_sampling_blockers(params)
                else "processed"
            )
            semantics.append(
                SpeculativeRequestSemantics(
                    request_id=int(request_id),
                    sampling_mode=sampling_mode,
                    mode="verify_chain",
                    context_tokens=max(1, int(request.context_len)),
                    remaining_decode=int(request.remaining_decode),
                )
            )
        resolve_capability = getattr(self.runner, "speculative_capability", None)
        capability = (
            resolve_capability(tuple(semantics))
            if callable(resolve_capability)
            else None
        )
        if capability is not None and not isinstance(capability, SpeculativeCapability):
            raise TypeError("runner speculative_capability must return SpeculativeCapability or None")
        self._speculative_cycle_sequence += 1
        operation_id = f"specdec2-cycle:{self._speculative_cycle_sequence}"
        graph_available = self._speculative_runner_flag(
            "speculative_graph_available",
            work,
            default=True,
        )
        target_available = self._speculative_runner_flag(
            "speculative_target_physical_available",
            work,
            default=True,
        )
        plan = plan_speculative_requests(
            capability,
            tuple(semantics),
            resident_slots=work.slot_ids,
            desired_candidate_counts=desired,
            operation_id=operation_id,
            cycle_id=self._speculative_cycle_sequence,
            context_bucket_size=self.scheduler.context_bucket_size,
            graph_available=graph_available,
            target_physical_available=target_available,
            suppress_speculation=suppression,
            declared_logical_c=work.declared_logical_c,
        )
        claims_fit = getattr(self.runner, "speculative_claims_fit", None)
        if plan.has_speculative_rows and callable(claims_fit) and not bool(claims_fit(plan)):
            plan = plan_speculative_requests(
                capability,
                tuple(semantics),
                resident_slots=work.slot_ids,
                desired_candidate_counts=desired,
                operation_id=operation_id,
                cycle_id=self._speculative_cycle_sequence,
                context_bucket_size=self.scheduler.context_bucket_size,
                claims_fit=False,
                graph_available=graph_available,
                target_physical_available=target_available,
                suppress_speculation=suppression,
                declared_logical_c=work.declared_logical_c,
            )
        self._last_speculative_plan = plan
        self._recent_speculative_plans.append(plan)
        if plan.is_ar_only:
            prepare_k0 = getattr(self.runner, "prepare_speculative_k0", None)
            if callable(prepare_k0):
                prepare_k0(plan, tuple(semantics), stream=None)
            return None
        start = time.perf_counter()
        try:
            result = self._run_staged_speculative_cycle(
                plan,
                tuple(semantics),
                capability,
            )
            if result is None:
                execute = getattr(self.runner, "execute_speculative_cycle", None)
                if not callable(execute):
                    return None
                result = execute(plan, commit=True)
        except BaseException as exc:
            recover = getattr(
                self.runner,
                "recover_speculative_cycle_failure",
                None,
            )
            if not callable(recover) or not bool(recover(plan, exc)):
                raise
            return None
        elapsed = time.perf_counter() - start
        if not isinstance(result, SpecCycleResult):
            raise TypeError("execute_speculative_cycle must return SpecCycleResult")
        if result.stage not in {
            SpecCycleStage.COMMITTED,
            SpecCycleStage.CANCELLED,
        }:
            raise RuntimeError("runner returned a non-terminal speculative cycle")
        if result.transaction.request_ids != plan.request_ids:
            raise ValueError("speculative result request_ids must match plan")
        if result.transaction.cycle_id != plan.cycle_id:
            raise ValueError("speculative result cycle_id must match plan")
        row_to_request = tuple(
            request_id
            for request_id, count in zip(
                plan.request_ids, plan.candidate_counts, strict=True
            )
            for _ in range(count)
        )
        spec_work = WorkItem(
            kind=WorkKind(plan.mode),
            request_ids=plan.request_ids,
            row_to_request=row_to_request,
            draft_depth=plan.max_candidate_count,
            slot_ids=work.slot_ids,
            active_mask=work.active_mask,
        )
        self.scheduler.record_work_duration(spec_work, elapsed)
        if result.stage is SpecCycleStage.CANCELLED:
            return self._cancelled_speculative_events(spec_work, result)
        pending = self._pending_speculative_cancellations(plan.request_ids)
        if pending:
            raise RuntimeError(
                "runner committed speculative output despite pending cancellation"
            )
        generated_events = self.scheduler.record_speculative_cycle_result(result)
        decorate_stream = getattr(
            self.runner,
            "decorate_speculative_stream_events",
            None,
        )
        if callable(decorate_stream):
            generated_events = tuple(decorate_stream(generated_events))
        return self._decode_events(spec_work, generated_events)

    def _pending_speculative_cancellations(
        self,
        request_ids: Sequence[int],
    ) -> tuple[int, ...]:
        cancelled: list[int] = []
        for request_id in request_ids:
            probe = self._speculative_cancel_probes.get(int(request_id))
            if probe is not None and bool(probe()):
                cancelled.append(int(request_id))
        return tuple(cancelled)

    def _cancelled_speculative_events(
        self,
        work: WorkItem,
        result: SpecCycleResult,
    ) -> tuple[EngineLoopEvent, ...]:
        self._last_work_kind = work.kind
        self._consecutive_prefill_chunks = 0
        self._cold_prefill_cohort_request_ids = frozenset()
        events: list[EngineLoopEvent] = [
            EngineLoopEvent(
                kind="work",
                request_ids=work.request_ids,
                work_kind=work.kind,
            )
        ]
        for request_id in result.cancelled_request_ids:
            completed = self.scheduler.cancel(int(request_id), reason="cancel")
            if completed is not None:
                events.append(
                    EngineLoopEvent(
                        kind="completed",
                        request_id=completed.request_id,
                        request_ids=(completed.request_id,),
                        completed=completed,
                    )
                )
        return tuple(events)

    def _run_staged_speculative_cycle(
        self,
        plan,
        semantics: tuple[SpeculativeRequestSemantics, ...],
        capability: SpeculativeCapability,
    ) -> SpecCycleResult | None:
        frontier_available = getattr(
            self.runner,
            "speculative_frontier_available",
            None,
        )
        if callable(frontier_available) and not bool(frontier_available(plan)):
            return None
        component_claims = getattr(self.runner, "speculative_component_claims", None)
        reserve_claims = getattr(self.runner, "reserve_speculative_claims", None)
        release_claims = getattr(self.runner, "release_speculative_claims", None)
        prepare = getattr(self.runner, "prepare_speculative_requests", None)
        propose = getattr(self.runner, "propose_speculative_batch", None)
        execute_target = getattr(self.runner, "execute_target_frontier", None)
        required = (
            component_claims,
            reserve_claims,
            release_claims,
            prepare,
            propose,
            execute_target,
        )
        if not all(callable(method) for method in required):
            return None
        components = component_claims(plan)
        complete_claims = compose_speculative_claims(plan.operation_id, components)
        reservation = reserve_claims(complete_claims)
        candidate_graph: CandidateGraph | None = None
        try:
            prepare(plan, semantics, stream=None)
            candidate_graph = propose(plan, semantics, stream=None)
            if not isinstance(candidate_graph, CandidateGraph):
                raise TypeError("propose_speculative_batch must return CandidateGraph")
            if candidate_graph.request_ids != plan.request_ids:
                raise ValueError("candidate graph request_ids must match plan")
            if candidate_graph.resident_slots != plan.resident_slots:
                raise ValueError("candidate graph resident_slots must match plan")
            if candidate_graph.candidate_counts != plan.candidate_counts:
                raise ValueError("candidate graph counts must match plan")
            root_tokens: list[int] = []
            root_positions: list[int] = []
            for request_id in plan.request_ids:
                request = self.scheduler.active_batch.requests[int(request_id)]
                root_tokens.append(
                    int(request.generated_tokens[-1])
                    if request.generated_tokens
                    else int(request.prompt_tokens[-1])
                )
                root_positions.append(int(request.context_len) - 1)
            if tuple(root_positions) != candidate_graph.root_positions:
                raise ValueError("candidate graph root positions must match scheduler state")
            live_spans_owner = getattr(
                self.runner,
                "speculative_kv_live_spans_owner",
                None,
            )
            owner = (
                live_spans_owner(plan)
                if callable(live_spans_owner)
                else f"{plan.operation_id}:target-live-spans"
            )
            frontier = TargetFrontier.from_candidate_graph(
                operation_id=plan.operation_id,
                candidate_graph=candidate_graph,
                root_tokens=tuple(root_tokens),
                physical_row_decomposition=plan.target_row_decomposition,
                transaction_mode=plan.target_transaction_mode,
                kv_storage_view_key=capability.kv_backend_key,
                kv_live_spans_owner=str(owner),
                execution_route=plan.execution_route,
            )
            result = execute_target(
                plan,
                frontier,
                complete_claims,
                commit=True,
                cancelled_request_ids=lambda: self._pending_speculative_cancellations(
                    plan.request_ids
                ),
            )
            if not isinstance(result, SpecCycleResult):
                raise TypeError("execute_target_frontier must return SpecCycleResult")
            if result.transaction.reserved_claims != complete_claims:
                raise ValueError("target result must own the complete speculative claims")
            return result
        except BaseException as exc:
            rollback = getattr(self.runner, "rollback_speculative_cycle", None)
            if callable(rollback):
                rollback(plan, candidate_graph, exc)
            raise
        finally:
            release_claims(reservation)

    def _speculative_runner_flag(
        self,
        name: str,
        work: WorkItem,
        *,
        default: bool,
    ) -> bool:
        value = getattr(self.runner, name, default)
        return bool(value(work) if callable(value) else value)

    def _decode_events(self, work: WorkItem, generated_events) -> tuple[EngineLoopEvent, ...]:
        self._last_work_kind = work.kind
        self._consecutive_prefill_chunks = 0
        self._cold_prefill_cohort_request_ids = frozenset()
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
        self._speculative_candidate_counts.pop(int(completed.request_id), None)
        self._speculative_cancel_probes.pop(int(completed.request_id), None)
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
    "GenerationAdmissionRejected",
    "GenerationSubmission",
    "EngineLoopRunner",
    "PREFILL_DECODE_POLICIES",
    "ResidentEngineLoop",
    "SubmitPollTextGenerator",
    "add_engine_loop_config_args",
    "engine_loop_config_from_args",
    "engine_loop_config_from_env",
]
