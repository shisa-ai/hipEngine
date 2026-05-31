"""Batch-friendly generation scheduler shell.

The scheduler is model-agnostic: it owns request ids, pending/admitted queues,
physical batch slots, prefill/decode work items, and completion routing.  Model
runners consume the emitted ``WorkItem`` metadata and report generated tokens
back through ``record_generated``.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from numbers import Integral
from typing import Iterable, Mapping, Sequence

from hipengine.dispatch import ActiveBatch, BatchShapeKey, RequestState, WorkItem, WorkKind
from hipengine.kvcache import KVTransaction
from hipengine.speculative import DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, TargetVerifyBuffers


@dataclass(frozen=True, slots=True)
class BatchGenerateRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int

    @classmethod
    def from_tokens(cls, request_id: int, prompt_tokens: Iterable[int], *, max_new_tokens: int) -> "BatchGenerateRequest":
        return cls(request_id=int(request_id), prompt_tokens=tuple(int(token) for token in prompt_tokens), max_new_tokens=int(max_new_tokens))


@dataclass(frozen=True, slots=True)
class PerRowSamplingParams:
    """Torch-free per-request sampling row for native sampler launches."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        stops = tuple(int(token) for token in self.stop_tokens)
        if any(token < 0 for token in stops):
            raise ValueError("stop_tokens must be non-negative")
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "top_p", float(self.top_p))
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "stop_tokens", stops)

    def resolved_seed(self, *, request_id: int, row_index: int) -> int:
        base = int(self.seed) if self.seed is not None else 0
        return _stable_sampler_seed(base_seed=base, request_id=int(request_id), row_index=int(row_index))


@dataclass(frozen=True, slots=True)
class SamplerParamsBlock:
    """Columnar per-row sampler params aligned with a decode work item."""

    request_ids: tuple[int, ...]
    temperatures: tuple[float, ...]
    top_ks: tuple[int, ...]
    top_ps: tuple[float, ...]
    repetition_penalties: tuple[float, ...]
    seeds: tuple[int, ...]
    stop_token_rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        rows = len(self.request_ids)
        if rows <= 0:
            raise ValueError("sampler params block must include at least one row")
        _check_len("temperatures", self.temperatures, rows)
        _check_len("top_ks", self.top_ks, rows)
        _check_len("top_ps", self.top_ps, rows)
        _check_len("repetition_penalties", self.repetition_penalties, rows)
        _check_len("seeds", self.seeds, rows)
        _check_len("stop_token_rows", self.stop_token_rows, rows)
        if len(set(self.request_ids)) != rows:
            raise ValueError("sampler params block request_ids must be unique")

    @classmethod
    def from_rows(
        cls,
        request_ids: Sequence[int],
        rows: Mapping[int, PerRowSamplingParams],
    ) -> "SamplerParamsBlock":
        ids = tuple(int(request_id) for request_id in request_ids)
        params = tuple(rows[request_id] for request_id in ids)
        return cls(
            request_ids=ids,
            temperatures=tuple(row.temperature for row in params),
            top_ks=tuple(row.top_k for row in params),
            top_ps=tuple(row.top_p for row in params),
            repetition_penalties=tuple(row.repetition_penalty for row in params),
            seeds=tuple(row.resolved_seed(request_id=request_id, row_index=index) for index, (request_id, row) in enumerate(zip(ids, params, strict=True))),
            stop_token_rows=tuple(row.stop_tokens for row in params),
        )

    def params_for(self, request_id: int) -> PerRowSamplingParams:
        index = self.request_ids.index(int(request_id))
        return PerRowSamplingParams(
            temperature=self.temperatures[index],
            top_k=self.top_ks[index],
            top_p=self.top_ps[index],
            repetition_penalty=self.repetition_penalties[index],
            seed=self.seeds[index],
            stop_tokens=self.stop_token_rows[index],
        )


@dataclass(frozen=True, slots=True)
class GeneratedToken:
    request_id: int
    token_id: int
    finished: bool = False


@dataclass(frozen=True, slots=True)
class RequestObservability:
    """Per-request timing/KV fields emitted with completion metadata."""

    queue_seconds: float
    prefill_seconds: float
    decode_seconds: float
    kv_pages_owned: int
    kv_pages_peak: int
    bucket_key: str | None
    admission_blocked_reason: str | None
    finish_reason: str
    submitted_timestamp: float
    admitted_timestamp: float | None
    completion_timestamp: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "queue_seconds": self.queue_seconds,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "kv_pages_owned": self.kv_pages_owned,
            "kv_pages_peak": self.kv_pages_peak,
            "bucket_key": self.bucket_key,
            "admission_blocked_reason": self.admission_blocked_reason,
            "finish_reason": self.finish_reason,
            "submitted_timestamp": self.submitted_timestamp,
            "admitted_timestamp": self.admitted_timestamp,
            "completion_timestamp": self.completion_timestamp,
        }


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    finished: bool
    finish_reason: str
    observability: RequestObservability

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": list(self.prompt_tokens),
            "generated_tokens": list(self.generated_tokens),
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "observability": self.observability.to_json_dict(),
        }


@dataclass(slots=True)
class _RequestObservabilityState:
    submitted_at: float
    admitted_at: float | None = None
    queue_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    kv_pages_owned: int = 0
    kv_pages_peak: int = 0
    bucket_key: str | None = None
    admission_blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CompactPromptBucket:
    """Scheduler bucket of prefill requests sharing one block-table length."""

    block_count: int
    request_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.block_count <= 0:
            raise ValueError("block_count must be positive")
        if not self.request_ids:
            raise ValueError("compact prompt bucket must include request_ids")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("compact prompt bucket request_ids must be unique")


@dataclass(frozen=True, slots=True)
class CompactPromptSlab:
    """Host compact-prompt slab descriptor for native c>N prefill.

    Runtime code materializes these tuples as device tensors before launching
    kernels. ``cu_seqlens_q``/``cu_seqlens_k`` define the block-diagonal prompt
    segments; ``block_tables`` is row-shaped because the current KV writer ABI
    requires a uniform block-table length for every row in one launch.
    """

    request_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    cu_seqlens_q: tuple[int, ...]
    cu_seqlens_k: tuple[int, ...]
    row_to_request: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]
    append_counts: tuple[int, ...]
    context_counts: tuple[int, ...]
    token_rows: tuple[tuple[int, ...], ...]
    block_count: int
    block_size: int = 256
    slot_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        rows = len(self.token_ids)
        if not self.request_ids:
            raise ValueError("compact prompt slab must include request_ids")
        if rows <= 0:
            raise ValueError("compact prompt slab must include token rows")
        if self.block_count <= 0:
            raise ValueError("block_count must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        _check_len("positions", self.positions, rows)
        _check_len("row_to_request", self.row_to_request, rows)
        _check_len("append_counts", self.append_counts, rows)
        _check_len("context_counts", self.context_counts, rows)
        _check_len("block_tables", self.block_tables, rows)
        _check_len("token_rows", self.token_rows, len(self.request_ids))
        if self.slot_ids:
            _check_len("slot_ids", self.slot_ids, len(self.request_ids))
            if len(set(self.slot_ids)) != len(self.slot_ids):
                raise ValueError("compact prompt slab slot_ids must be unique")
            if any(slot < 0 for slot in self.slot_ids):
                raise ValueError("compact prompt slab slot_ids must be non-negative")
        _check_len("cu_seqlens_q", self.cu_seqlens_q, len(self.request_ids) + 1)
        _check_len("cu_seqlens_k", self.cu_seqlens_k, len(self.request_ids) + 1)
        if self.cu_seqlens_q[0] != 0 or self.cu_seqlens_k[0] != 0:
            raise ValueError("cu_seqlens must start at 0")
        if self.cu_seqlens_q[-1] != rows or self.cu_seqlens_k[-1] != rows:
            raise ValueError("cu_seqlens must end at total row count")
        if any(a > b for a, b in zip(self.cu_seqlens_q, self.cu_seqlens_q[1:])):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        if any(a > b for a, b in zip(self.cu_seqlens_k, self.cu_seqlens_k[1:])):
            raise ValueError("cu_seqlens_k must be non-decreasing")
        if set(self.row_to_request).difference(self.request_ids):
            raise ValueError("row_to_request contains request id outside request_ids")
        if any(len(row) != self.block_count for row in self.block_tables):
            raise ValueError("block_tables rows must match block_count")
        if any(position < 0 for position in self.positions):
            raise ValueError("positions must be non-negative")
        if any(count < 0 for count in self.append_counts):
            raise ValueError("append_counts must be non-negative")
        if any(count <= 0 for count in self.context_counts):
            raise ValueError("context_counts must be positive")

    @classmethod
    def from_token_rows(
        cls,
        *,
        request_ids: Sequence[int],
        token_rows: Sequence[Sequence[int]],
        start_positions: Sequence[int],
        block_count: int,
        block_size: int = 256,
        block_tables_by_request: Sequence[Sequence[int]] | None = None,
        slot_ids: Sequence[int] | None = None,
    ) -> "CompactPromptSlab":
        request_tuple = tuple(int(request_id) for request_id in request_ids)
        row_tuple = tuple(tuple(int(token) for token in row) for row in token_rows)
        starts = tuple(int(position) for position in start_positions)
        if len(request_tuple) != len(row_tuple) or len(request_tuple) != len(starts):
            raise ValueError("request_ids, token_rows, and start_positions must align")
        slot_tuple = () if slot_ids is None else tuple(int(slot) for slot in slot_ids)
        if slot_tuple and len(slot_tuple) != len(request_tuple):
            raise ValueError("slot_ids must align with request_ids")
        if block_tables_by_request is None:
            request_tables = tuple(tuple(range(int(block_count))) for _ in request_tuple)
        else:
            request_tables = tuple(tuple(int(block) for block in table) for table in block_tables_by_request)
            if len(request_tables) != len(request_tuple):
                raise ValueError("block_tables_by_request must align with request_ids")
        token_ids: list[int] = []
        positions: list[int] = []
        row_to_request: list[int] = []
        block_tables: list[tuple[int, ...]] = []
        cu = [0]
        for request_id, tokens, start, table in zip(request_tuple, row_tuple, starts, request_tables, strict=True):
            if not tokens:
                raise ValueError("compact prompt token rows must be non-empty")
            for offset, token in enumerate(tokens):
                token_ids.append(token)
                positions.append(start + offset)
                row_to_request.append(request_id)
                block_tables.append(table)
            cu.append(len(token_ids))
        return cls(
            request_ids=request_tuple,
            token_ids=tuple(token_ids),
            positions=tuple(positions),
            cu_seqlens_q=tuple(cu),
            cu_seqlens_k=tuple(cu),
            row_to_request=tuple(row_to_request),
            block_tables=tuple(block_tables),
            append_counts=tuple(positions),
            context_counts=tuple(position + 1 for position in positions),
            token_rows=row_tuple,
            block_count=int(block_count),
            block_size=int(block_size),
            slot_ids=slot_tuple,
        )

    @property
    def rows(self) -> int:
        return len(self.token_ids)

    @property
    def request_count(self) -> int:
        return len(self.request_ids)

    @property
    def physical_slot_ids(self) -> tuple[int, ...]:
        """Physical slot ids for runtime commit, defaulting to request ids for old fixtures."""

        return self.slot_ids or self.request_ids

    def to_work_item(self) -> WorkItem:
        return WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=self.request_ids,
            row_to_request=self.row_to_request,
            token_rows=self.token_rows,
        )


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyWork:
    target_batch: TargetVerifyBatch
    work_item: WorkItem


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyPlan:
    target_batch: TargetVerifyBatch
    work_item: WorkItem
    transaction: KVTransaction
    shape_key: BatchShapeKey
    graph: object


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyBufferPlan:
    plan: SpeculativeVerifyPlan
    buffers: TargetVerifyBuffers


@dataclass(frozen=True, slots=True)
class SpeculativeCommitPlan:
    verify_plan: SpeculativeVerifyBufferPlan
    summary: TargetAcceptSummary
    commit_plan: TargetCommitPlan


@dataclass(frozen=True, slots=True)
class SpeculativeStateCommitPlan:
    commit_plan: SpeculativeCommitPlan
    buffers: TargetStateCommitBuffers


GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS = ("le_10us", "le_100us", "le_1ms", "le_10ms", "gt_10ms")


@dataclass(frozen=True, slots=True)
class GraphBucketStats:
    entries: int
    hits: int
    misses: int
    miss_reasons: Mapping[str, int] = field(default_factory=dict)
    kernel_time_histogram_ns: Mapping[str, int] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        lookup_count = int(self.hits) + int(self.misses)
        replay_hit_rate = float(self.hits) / float(lookup_count) if lookup_count > 0 else 0.0
        kernel_time_histogram = {bucket: 0 for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS}
        for key, value in self.kernel_time_histogram_ns.items():
            if key in kernel_time_histogram:
                kernel_time_histogram[str(key)] = int(value)
        return {
            "entries": int(self.entries),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "replay_hit_rate": replay_hit_rate,
            "miss_reasons": {str(key): int(value) for key, value in sorted(self.miss_reasons.items())},
            "kernel_time_histogram_ns": kernel_time_histogram,
        }


class GraphBucketCache:
    """Tiny graph bucket cache keyed by full batch/specdecode shape."""

    def __init__(self) -> None:
        self._cache: dict[BatchShapeKey, object] = {}
        self._hits = 0
        self._misses = 0
        self._miss_reasons: Counter[str] = Counter()
        self._kernel_time_histogram_ns: Counter[str] = Counter()

    @property
    def stats(self) -> GraphBucketStats:
        return GraphBucketStats(
            entries=len(self._cache),
            hits=self._hits,
            misses=self._misses,
            miss_reasons=dict(self._miss_reasons),
            kernel_time_histogram_ns=dict(self._kernel_time_histogram_ns),
        )

    def get(self, key: BatchShapeKey, *, miss_reason: str = "cache_absent") -> object | None:
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        self._miss_reasons[str(miss_reason or "unspecified")] += 1
        return None

    def put(self, key: BatchShapeKey, graph: object) -> None:
        self._cache[key] = graph

    def get_or_create(self, key: BatchShapeKey, factory, *, miss_reason: str = "cache_absent") -> object:
        cached = self.get(key, miss_reason=miss_reason)
        if cached is not None:
            return cached
        graph = factory(key)
        self.put(key, graph)
        return graph

    def record_kernel_time_ns(self, duration_ns: int) -> None:
        if not isinstance(duration_ns, Integral) or isinstance(duration_ns, bool):
            raise ValueError("duration_ns must be a non-negative integer")
        ns = int(duration_ns)
        if ns < 0:
            raise ValueError("duration_ns must be a non-negative integer")
        self._kernel_time_histogram_ns[_kernel_time_histogram_bucket_ns(ns)] += 1

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._miss_reasons.clear()
        self._kernel_time_histogram_ns.clear()


def _kernel_time_histogram_bucket_ns(duration_ns: int) -> str:
    if duration_ns <= 10_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[0]
    if duration_ns <= 100_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[1]
    if duration_ns <= 1_000_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[2]
    if duration_ns <= 10_000_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[3]
    return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[4]


class ResidentBatchScheduler:
    """Continuous-batching scheduler shell for resident decode runners."""

    def __init__(
        self,
        *,
        capacity: int,
        context_bucket_size: int = 256,
        clock: Callable[[], float] | None = None,
        reclaim_callback: Callable[[CompletedRequest], None] | None = None,
        max_pending_requests: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if context_bucket_size <= 0:
            raise ValueError("context_bucket_size must be positive")
        if max_pending_requests is not None and max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be positive when set")
        self.capacity = int(capacity)
        self.context_bucket_size = int(context_bucket_size)
        self.max_pending_requests = None if max_pending_requests is None else int(max_pending_requests)
        self.active_batch = ActiveBatch(self.capacity)
        self.graph_buckets = GraphBucketCache()
        self._pending: deque[RequestState] = deque()
        self._completed: dict[int, CompletedRequest] = {}
        self._observability: dict[int, _RequestObservabilityState] = {}
        self._sampling: dict[int, PerRowSamplingParams] = {}
        self._next_request_id = 0
        self._clock = time.monotonic if clock is None else clock
        self._reclaim_callback = reclaim_callback

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return self.active_batch.active_count

    @property
    def completed(self) -> Mapping[int, CompletedRequest]:
        return self._completed

    def submit(
        self,
        prompt_tokens: Iterable[int],
        *,
        max_new_tokens: int,
        request_id: int | None = None,
        sampling: PerRowSamplingParams | None = None,
    ) -> int:
        if self.max_pending_requests is not None and len(self._pending) >= self.max_pending_requests:
            raise ValueError(f"pending request queue is full (max_pending_requests={self.max_pending_requests})")
        rid = self._allocate_request_id() if request_id is None else int(request_id)
        if rid in self.active_batch.requests or any(req.request_id == rid for req in self._pending) or rid in self._completed:
            raise ValueError(f"request_id {rid} already exists")
        self._pending.append(RequestState.from_tokens(rid, prompt_tokens, max_new_tokens=max_new_tokens))
        self._sampling[rid] = sampling or PerRowSamplingParams()
        self._observability[rid] = _RequestObservabilityState(
            submitted_at=self._clock(),
            admission_blocked_reason="capacity" if self.active_batch.active_count >= self.capacity else None,
        )
        return rid

    def admit_pending(self) -> tuple[int, ...]:
        """Fill free slots from the pending queue and return admitted request ids."""

        admitted: list[int] = []
        while self._pending and self.active_batch.active_count < self.capacity:
            request = self._pending.popleft()
            self.active_batch.admit(request)
            state = self._observability.get(request.request_id)
            if state is not None:
                now = self._clock()
                state.admitted_at = now
                state.queue_seconds = max(0.0, now - state.submitted_at)
            admitted.append(request.request_id)
        if self._pending and self.active_batch.active_count >= self.capacity:
            for request in self._pending:
                state = self._observability.get(request.request_id)
                if state is not None and state.admission_blocked_reason is None:
                    state.admission_blocked_reason = "capacity"
        return tuple(admitted)

    def compact(self, order: Sequence[int] | None = None):
        return self.active_batch.compact(order=order)

    def next_prefill_work(self, *, chunk_size: int) -> WorkItem | None:
        """Emit one prefill chunk and advance the request's prompt cursor."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for request_id in self.active_batch.active_request_ids:
            request = self.active_batch.requests[request_id]
            if request.remaining_prefill <= 0:
                continue
            updated, chunk = request.take_prefill(chunk_size)
            self.active_batch.update_request(updated)
            self._update_kv_pages(updated)
            self._set_bucket_key((request_id,), self._bucket_key(self.shape_key(mode=WorkKind.PREFILL)))
            return WorkItem(
                kind=WorkKind.PREFILL,
                request_ids=(request_id,),
                row_to_request=(request_id,),
                token_rows=(chunk,),
            )
        return None

    def bucketize_by_block_count(
        self,
        *,
        chunk_size: int,
        block_size: int = 256,
        request_ids: Sequence[int] | None = None,
    ) -> tuple[CompactPromptBucket, ...]:
        """Group active prefill requests by the KV block count needed now.

        The compact KV writer currently requires one block-table length per
        launch. This host bucketization is the guardrail that prevents silently
        mixing requests with different per-request block-table lengths.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        candidate_ids = self.active_batch.active_request_ids if request_ids is None else tuple(int(item) for item in request_ids)
        buckets: dict[int, list[int]] = {}
        for request_id in candidate_ids:
            if request_id not in self.active_batch.requests:
                raise KeyError(request_id)
            request = self.active_batch.requests[request_id]
            if request.finished or request.remaining_prefill <= 0:
                continue
            rows = min(int(chunk_size), request.remaining_prefill)
            end_position_exclusive = request.next_prompt_index + rows
            block_count = max(1, ceil(end_position_exclusive / int(block_size)))
            buckets.setdefault(block_count, []).append(request_id)
        return tuple(
            CompactPromptBucket(block_count=block_count, request_ids=tuple(ids))
            for block_count, ids in sorted(buckets.items())
        )

    def next_compact_prefill_slabs(
        self,
        *,
        chunk_size: int,
        block_size: int = 256,
    ) -> tuple[CompactPromptSlab, ...]:
        """Emit compact c>N prefill slab descriptors and advance cursors.

        Each returned slab contains requests with a common block-table length.
        Runtime code must execute each slab natively or reject it explicitly;
        this scheduler method does not fall back to per-request prompt loops.
        """

        slabs: list[CompactPromptSlab] = []
        for bucket in self.bucketize_by_block_count(chunk_size=chunk_size, block_size=block_size):
            token_rows: list[tuple[int, ...]] = []
            start_positions: list[int] = []
            for request_id in bucket.request_ids:
                request = self.active_batch.requests[request_id]
                start_positions.append(request.next_prompt_index)
                updated, chunk = request.take_prefill(chunk_size)
                self.active_batch.update_request(updated)
                self._update_kv_pages(updated, block_size=block_size)
                token_rows.append(chunk)
            slab = CompactPromptSlab.from_token_rows(
                request_ids=bucket.request_ids,
                token_rows=token_rows,
                start_positions=start_positions,
                block_count=bucket.block_count,
                block_size=block_size,
                slot_ids=tuple(self.active_batch.slot_for(request_id) for request_id in bucket.request_ids),
            )
            self._set_bucket_key(
                bucket.request_ids,
                f"prefill:compact:blocks={slab.block_count}:rows={slab.rows}:block_size={slab.block_size}",
            )
            slabs.append(slab)
        return tuple(slabs)

    def has_prefill_work(self) -> bool:
        """Return whether any active request still needs prompt prefill."""

        return any(
            request.remaining_prefill > 0 and not request.finished
            for request in self.active_batch.requests.values()
        )

    def next_decode_work(self) -> WorkItem | None:
        """Emit one decode step over active requests with completed prefill."""

        request_ids = tuple(
            request_id
            for request_id in self.active_batch.active_request_ids
            if self.active_batch.requests[request_id].remaining_prefill == 0
            and self.active_batch.requests[request_id].remaining_decode > 0
            and not self.active_batch.requests[request_id].finished
        )
        if not request_ids:
            return None
        self._set_bucket_key(request_ids, self._bucket_key(self.shape_key(mode=WorkKind.DECODE)))
        return WorkItem(kind=WorkKind.DECODE, request_ids=request_ids, row_to_request=request_ids)

    def sampler_params_block(self, request_ids: Sequence[int]) -> SamplerParamsBlock:
        """Return a columnar per-row sampler block for a native decode launch."""

        ids = tuple(int(request_id) for request_id in request_ids)
        for request_id in ids:
            if request_id not in self._sampling:
                raise KeyError(f"no sampler params for request_id {request_id}")
        return SamplerParamsBlock.from_rows(ids, self._sampling)

    def record_work_duration(self, work: WorkItem, seconds: float) -> None:
        """Attach measured runner time to the request observability rows."""

        elapsed = float(seconds)
        if elapsed < 0.0:
            raise ValueError("work duration must be non-negative")
        if work.kind is WorkKind.PREFILL:
            field = "prefill_seconds"
        elif work.kind is WorkKind.DECODE:
            field = "decode_seconds"
        else:
            return
        for request_id in work.request_ids:
            state = self._observability.get(request_id)
            if state is None:
                continue
            if field == "prefill_seconds":
                state.prefill_seconds += elapsed
            elif field == "decode_seconds":
                state.decode_seconds += elapsed

    def next_speculative_verify_work(
        self,
        draft: DraftBatch,
        *,
        root_tokens: Sequence[int],
        root_positions: Sequence[int],
    ) -> SpeculativeVerifyWork:
        """Emit scheduler metadata for a target verification batch.

        This only validates scheduler ownership/readiness and materializes the
        root+candidate row layout.  It does not run a draft model, target
        verifier, or commit accepted state.
        """

        for request_id in draft.request_ids:
            if request_id not in self.active_batch.requests:
                raise KeyError(request_id)
            request = self.active_batch.requests[request_id]
            if request.remaining_prefill != 0:
                raise ValueError("speculative verification requires completed prefill")
            if request.remaining_decode <= 0 or request.finished:
                raise ValueError("speculative verification requires an active decode request")
        target = TargetVerifyBatch.from_draft(draft, root_tokens=root_tokens, root_positions=root_positions)
        return SpeculativeVerifyWork(target_batch=target, work_item=target.to_work_item())

    def begin_speculative_verify_transaction(self, kv_policy, work: SpeculativeVerifyWork):
        """Begin the KV transaction for scheduler-owned target verification."""

        if work.work_item.request_ids != work.target_batch.request_ids:
            raise ValueError("speculative work request_ids must match target batch")
        seqs = tuple(self.active_batch.requests[request_id] for request_id in work.target_batch.request_ids)
        return kv_policy.begin_transaction(seqs, work.target_batch)

    def record_generated(self, tokens: Sequence[GeneratedToken | tuple[int, int] | tuple[int, int, bool]]) -> tuple[CompletedRequest, ...]:
        """Record generated tokens and reclaim newly completed requests."""

        completed: list[CompletedRequest] = []
        for item in tokens:
            token = _coerce_generated_token(item)
            done = self._append_generated_token(token)
            if done is not None:
                completed.append(done)
        return tuple(completed)

    def cancel(self, request_id: int, *, reason: str = "cancel") -> CompletedRequest | None:
        """Cancel a pending or active request through the unified reclaim path."""

        if reason not in {"cancel", "disconnect", "timeout"}:
            raise ValueError("cancel reason must be cancel, disconnect, or timeout")
        rid = int(request_id)
        pending = self._pop_pending_request(rid)
        if pending is not None:
            return self._complete_request(pending, finish_reason=reason)
        if rid not in self.active_batch.requests:
            return None
        return self._reclaim_active_request(rid, finish_reason=reason)

    def disconnect(self, request_id: int) -> CompletedRequest | None:
        """Mark a client disconnect through the same reclaim path as cancel."""

        return self.cancel(request_id, reason="disconnect")

    def timeout(self, request_id: int) -> CompletedRequest | None:
        """Mark a per-request timeout through the same reclaim path as cancel."""

        return self.cancel(request_id, reason="timeout")

    def record_speculative_accept(self, summary: TargetAcceptSummary) -> tuple[CompletedRequest, ...]:
        """Record accepted speculative tokens plus optional target next tokens."""

        next_tokens = summary.next_tokens or (None,) * len(summary.request_ids)
        for request_id, tokens, next_token in zip(summary.request_ids, summary.accepted_tokens, next_tokens, strict=True):
            if request_id not in self.active_batch.requests:
                raise KeyError(request_id)
            request = self.active_batch.requests[request_id]
            output_count = len(tokens) + (0 if next_token is None else 1)
            if output_count > request.remaining_decode:
                raise ValueError("accepted speculative output tokens exceed remaining decode budget")
        completed: list[CompletedRequest] = []
        for request_id, tokens, next_token in zip(summary.request_ids, summary.accepted_tokens, next_tokens, strict=True):
            output_tokens = tokens if next_token is None else (*tokens, next_token)
            for token_id in output_tokens:
                done = self._append_generated_token(GeneratedToken(request_id, token_id))
                if done is not None:
                    completed.append(done)
                    break
        return tuple(completed)

    def speculative_verify_shape_key(
        self,
        work: SpeculativeVerifyWork,
        *,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> BatchShapeKey:
        """Return the graph bucket key for scheduler-owned verify work."""

        return work.target_batch.shape_key(
            self.active_batch,
            context_bucket_size=self.context_bucket_size,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )

    def get_or_create_speculative_verify_graph(
        self,
        work: SpeculativeVerifyWork,
        factory,
        *,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> object:
        """Cache graph/replay objects for scheduler-owned verify work."""

        key = self.speculative_verify_shape_key(
            work,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )
        return self.graph_buckets.get_or_create(key, factory)

    def plan_speculative_verify(
        self,
        kv_policy,
        work: SpeculativeVerifyWork,
        factory,
        *,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> SpeculativeVerifyPlan:
        """Bundle scheduler metadata for one native target-verifier replay."""

        transaction = self.begin_speculative_verify_transaction(kv_policy, work)
        if transaction.request_ids != work.target_batch.request_ids:
            raise ValueError("speculative transaction request_ids must match target batch")
        if transaction.draft_rows != work.target_batch.candidate_count:
            raise ValueError("speculative transaction rows must match target candidate rows")
        if transaction.candidate_counts is not None and transaction.candidate_counts != work.target_batch.candidate_counts:
            raise ValueError("speculative transaction candidate counts must match target batch")
        key = self.speculative_verify_shape_key(
            work,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )
        graph = self.graph_buckets.get_or_create(key, factory)
        return SpeculativeVerifyPlan(
            target_batch=work.target_batch,
            work_item=work.work_item,
            transaction=transaction,
            shape_key=key,
            graph=graph,
        )

    def bind_speculative_verify_buffers(
        self,
        plan: SpeculativeVerifyPlan,
        buffers: TargetVerifyBuffers,
    ) -> SpeculativeVerifyBufferPlan:
        """Bind target-verifier device buffers to a scheduler plan."""

        if buffers.request_ids != plan.target_batch.request_ids:
            raise ValueError("target verify buffers request_ids must match speculative plan")
        if buffers.rows != plan.target_batch.rows:
            raise ValueError("target verify buffer rows must match speculative plan")
        if buffers.candidate_rows != plan.target_batch.candidate_count:
            raise ValueError("target verify candidate rows must match speculative plan")
        if buffers.candidate_counts is not None and buffers.candidate_counts != plan.target_batch.candidate_counts:
            raise ValueError("target verify buffer candidate_counts must match speculative plan")
        if buffers.draft_depth is not None and buffers.draft_depth != plan.target_batch.draft_depth:
            raise ValueError("target verify buffer draft_depth must match speculative plan")
        if buffers.tree_shape is not None and buffers.tree_shape != plan.target_batch.tree_shape:
            raise ValueError("target verify buffer tree_shape must match speculative plan")
        if buffers.transaction_id is not None and buffers.transaction_id != plan.transaction.transaction_id:
            raise ValueError("target verify buffers transaction_id must match speculative plan")
        if buffers.mode != plan.target_batch.mode:
            raise ValueError("target verify buffer mode must match speculative plan")
        return SpeculativeVerifyBufferPlan(plan=plan, buffers=buffers)

    def plan_speculative_commit(
        self,
        verify_plan: SpeculativeVerifyBufferPlan,
        summary: TargetAcceptSummary,
    ) -> SpeculativeCommitPlan:
        """Build the scheduler-owned commit plan for accepted verify rows."""

        target = verify_plan.plan.target_batch
        if summary.request_ids != target.request_ids:
            raise ValueError("accept summary request_ids must match speculative plan")
        if summary.transaction_id is not None and summary.transaction_id != verify_plan.plan.transaction.transaction_id:
            raise ValueError("accept summary transaction_id must match speculative plan")
        if summary.mode != target.mode:
            raise ValueError("accept summary mode must match speculative plan")
        if summary.candidate_counts is not None and summary.candidate_counts != target.candidate_counts:
            raise ValueError("accept summary candidate_counts must match speculative plan")
        if summary.draft_depth is not None and summary.draft_depth != target.draft_depth:
            raise ValueError("accept summary draft_depth must match speculative plan")
        if summary.tree_shape is not None and summary.tree_shape != target.tree_shape:
            raise ValueError("accept summary tree_shape must match speculative plan")
        root_rows = set(target.root_rows)
        candidate_rows = set(target.candidate_rows)
        for request_id, count, row, token, position in zip(
            summary.request_ids,
            summary.accepted_counts,
            summary.commit_rows,
            summary.commit_tokens,
            summary.commit_positions,
            strict=True,
        ):
            if row < 0 or row >= target.rows:
                raise ValueError("accept summary commit row must be in target batch")
            if target.row_to_request[row] != request_id:
                raise ValueError("accept summary commit row must belong to its request")
            if count == 0 and row not in root_rows:
                raise ValueError("zero accepted candidates must commit the request root row")
            if count > 0 and row not in candidate_rows:
                raise ValueError("accepted candidates must commit a candidate row")
            if target.draft_depths[row] != count:
                raise ValueError("accept summary commit row depth must match accepted count")
            if target.tokens[row] != token or target.positions[row] != position:
                raise ValueError("accept summary commit token/position must match target row")
        commit = TargetCommitPlan.from_summary(summary, verify_plan.plan.transaction)
        return SpeculativeCommitPlan(verify_plan=verify_plan, summary=summary, commit_plan=commit)

    def plan_speculative_commit_from_top1(
        self,
        verify_plan: SpeculativeVerifyBufferPlan,
        target_top1: Sequence[int],
        *,
        remaining_decode: Sequence[int] | None = None,
    ) -> SpeculativeCommitPlan:
        """Build a scheduler commit plan from target top-1 row outputs."""

        target = verify_plan.plan.target_batch
        if remaining_decode is None:
            budgets = tuple(self.active_batch.requests[request_id].remaining_decode for request_id in target.request_ids)
        else:
            budgets = tuple(int(count) for count in remaining_decode)
        result = target.accept_from_top1(
            target_top1,
            transaction_id=verify_plan.plan.transaction.transaction_id,
            remaining_decode=budgets,
        )
        summary = TargetAcceptSummary.from_accept_result(target, result)
        return self.plan_speculative_commit(verify_plan, summary)

    def bind_speculative_commit_buffers(
        self,
        plan: SpeculativeCommitPlan,
        buffers: TargetStateCommitBuffers,
    ) -> SpeculativeStateCommitPlan:
        """Bind verified state/KV commit buffers to a scheduler commit plan."""

        commit = plan.commit_plan
        if buffers.request_ids != commit.request_ids:
            raise ValueError("state commit buffers request_ids must match speculative commit plan")
        if buffers.transaction_id != commit.transaction_id:
            raise ValueError("state commit buffers transaction_id must match speculative commit plan")
        if buffers.mode != commit.mode:
            raise ValueError("state commit buffers mode must match speculative commit plan")
        if buffers.device != plan.verify_plan.buffers.device:
            raise ValueError("state commit buffers must live on target verify device")
        if not buffers.has_linear_state and not buffers.has_kv_rows:
            raise ValueError("state commit buffers must include linear state or KV rows")
        target_rows = plan.verify_plan.plan.target_batch.rows
        accepted_rows = sum(commit.accepted_counts)
        if buffers.linear_state_src is not None and buffers.linear_state_src.shape[0] < target_rows:
            raise ValueError("linear state source rows must cover target verify rows")
        if buffers.kv_rows_src is not None and buffers.kv_rows_src.shape[0] < target_rows:
            raise ValueError("KV source rows must cover target verify rows")
        if buffers.kv_rows_dst is not None and buffers.kv_rows_dst.shape[0] < accepted_rows:
            raise ValueError("KV destination rows must cover accepted token rows")
        return SpeculativeStateCommitPlan(commit_plan=plan, buffers=buffers)

    def commit_speculative_kv_transaction(self, kv_policy, plan: SpeculativeStateCommitPlan) -> KVTransaction:
        """Mark the scheduler-owned speculative KV transaction committed."""

        commit = plan.commit_plan.commit_plan
        transaction = plan.commit_plan.verify_plan.plan.transaction
        if commit.transaction_id != transaction.transaction_id:
            raise ValueError("speculative commit plan transaction_id must match KV transaction")
        if commit.request_ids != transaction.request_ids:
            raise ValueError("speculative commit plan request_ids must match KV transaction")
        return kv_policy.commit(transaction, commit.kv_accept_counts)

    def rollback_speculative_kv_transaction(self, kv_policy, plan: SpeculativeVerifyPlan) -> KVTransaction:
        """Rollback a scheduler-owned speculative KV transaction."""

        transaction = plan.transaction
        if transaction.request_ids != plan.target_batch.request_ids:
            raise ValueError("speculative transaction request_ids must match target batch")
        if transaction.draft_rows != plan.target_batch.candidate_count:
            raise ValueError("speculative transaction rows must match target candidate rows")
        return kv_policy.rollback(transaction)

    def finalize_speculative_accept(
        self,
        committed_transaction: KVTransaction,
        plan: SpeculativeStateCommitPlan,
    ) -> tuple[CompletedRequest, ...]:
        """Record accepted tokens after the speculative KV transaction commits."""

        commit = plan.commit_plan.commit_plan
        if committed_transaction.transaction_id != commit.transaction_id:
            raise ValueError("committed KV transaction_id must match speculative commit plan")
        if committed_transaction.request_ids != commit.request_ids:
            raise ValueError("committed KV request_ids must match speculative commit plan")
        if not committed_transaction.committed or committed_transaction.rolled_back:
            raise ValueError("speculative KV transaction must be committed and not rolled back")
        if committed_transaction.accepted_counts != commit.kv_accept_counts:
            raise ValueError("committed KV accepted_counts must match speculative commit plan")
        return self.record_speculative_accept(plan.commit_plan.summary)

    def shape_key(self, *, mode: WorkKind | str, top_k: int = 0, experts_per_token: int = 0, replay_steps: int = 1) -> BatchShapeKey:
        return self.active_batch.shape_key(
            mode=mode,
            context_bucket_size=self.context_bucket_size,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )

    def _bucket_key(self, key: BatchShapeKey) -> str:
        mask = "".join("1" if active else "0" for active in key.active_mask)
        return (
            f"{key.mode.value}:c={key.active_c}:ctx={key.context_bucket}:mask={mask}:"
            f"top_k={key.top_k}:experts={key.experts_per_token}:replay={key.replay_steps}:"
            f"draft={key.draft_depth}"
        )

    def _set_bucket_key(self, request_ids: Sequence[int], bucket_key: str) -> None:
        for request_id in request_ids:
            state = self._observability.get(int(request_id))
            if state is not None:
                state.bucket_key = bucket_key

    def _update_kv_pages(self, request: RequestState, *, block_size: int = 256) -> None:
        state = self._observability.get(request.request_id)
        if state is None:
            return
        pages = max(0, ceil(request.context_len / int(block_size)))
        state.kv_pages_owned = pages
        state.kv_pages_peak = max(state.kv_pages_peak, pages)

    def _allocate_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _append_generated_token(self, token: GeneratedToken) -> CompletedRequest | None:
        request = self.active_batch.requests[token.request_id]
        finish_reason = "stop" if token.finished else "length"
        updated = request.append_generated(token.token_id, finished=token.finished)
        self.active_batch.update_request(updated)
        self._update_kv_pages(updated)
        if not updated.finished:
            return None
        return self._reclaim_active_request(updated.request_id, finish_reason=finish_reason)

    def _pop_pending_request(self, request_id: int) -> RequestState | None:
        for pending in tuple(self._pending):
            if pending.request_id == request_id:
                self._pending = deque(item for item in self._pending if item.request_id != request_id)
                return pending
        return None

    def _reclaim_active_request(self, request_id: int, *, finish_reason: str) -> CompletedRequest:
        self.active_batch.finish(request_id)
        reclaimed = self.active_batch.reclaim(request_id)
        return self._complete_request(reclaimed, finish_reason=finish_reason)

    def _complete_request(self, request: RequestState, *, finish_reason: str) -> CompletedRequest:
        now = self._clock()
        self._update_kv_pages(request)
        state = self._observability.pop(
            request.request_id,
            _RequestObservabilityState(submitted_at=now, admitted_at=now),
        )
        if state.admitted_at is None:
            state.queue_seconds = max(0.0, now - state.submitted_at)
        observability = RequestObservability(
            queue_seconds=state.queue_seconds,
            prefill_seconds=state.prefill_seconds,
            decode_seconds=state.decode_seconds,
            kv_pages_owned=state.kv_pages_owned,
            kv_pages_peak=state.kv_pages_peak,
            bucket_key=state.bucket_key,
            admission_blocked_reason=state.admission_blocked_reason,
            finish_reason=finish_reason,
            submitted_timestamp=state.submitted_at,
            admitted_timestamp=state.admitted_at,
            completion_timestamp=now,
        )
        done = CompletedRequest(
            request_id=request.request_id,
            prompt_tokens=request.prompt_tokens,
            generated_tokens=request.generated_tokens,
            finished=True,
            finish_reason=finish_reason,
            observability=observability,
        )
        self._sampling.pop(done.request_id, None)
        self._completed[done.request_id] = done
        if self._reclaim_callback is not None:
            self._reclaim_callback(done)
        return done


def _stable_sampler_seed(*, base_seed: int, request_id: int, row_index: int) -> int:
    """Derive a deterministic uint64-ish row seed without process-random hash()."""

    value = (int(base_seed) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value ^= (int(request_id) + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= (int(row_index) + 0xD6E8FEB86659FD93) & ((1 << 64) - 1)
    return value & ((1 << 63) - 1)


def _coerce_generated_token(item: GeneratedToken | tuple[int, int] | tuple[int, int, bool]) -> GeneratedToken:
    if isinstance(item, GeneratedToken):
        return item
    if len(item) == 2:
        request_id, token_id = item
        return GeneratedToken(int(request_id), int(token_id), False)
    request_id, token_id, finished = item
    return GeneratedToken(int(request_id), int(token_id), bool(finished))


def _check_len(name: str, value: Sequence[object], expected: int) -> None:
    if len(value) != expected:
        raise ValueError(f"{name} length must be {expected}")


__all__ = [
    "BatchGenerateRequest",
    "CompactPromptBucket",
    "CompactPromptSlab",
    "CompletedRequest",
    "GeneratedToken",
    "GraphBucketCache",
    "GraphBucketStats",
    "PerRowSamplingParams",
    "RequestObservability",
    "ResidentBatchScheduler",
    "SamplerParamsBlock",
    "SpeculativeCommitPlan",
    "SpeculativeStateCommitPlan",
    "SpeculativeVerifyBufferPlan",
    "SpeculativeVerifyPlan",
    "SpeculativeVerifyWork",
]
