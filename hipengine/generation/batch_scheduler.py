"""Batch-friendly generation scheduler shell.

The scheduler is model-agnostic: it owns request ids, pending/admitted queues,
physical batch slots, prefill/decode work items, and completion routing.  Model
runners consume the emitted ``WorkItem`` metadata and report generated tokens
back through ``record_generated``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
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
class GeneratedToken:
    request_id: int
    token_id: int
    finished: bool = False


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    finished: bool


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


@dataclass(frozen=True, slots=True)
class GraphBucketStats:
    entries: int
    hits: int
    misses: int


class GraphBucketCache:
    """Tiny graph bucket cache keyed by full batch/specdecode shape."""

    def __init__(self) -> None:
        self._cache: dict[BatchShapeKey, object] = {}
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> GraphBucketStats:
        return GraphBucketStats(entries=len(self._cache), hits=self._hits, misses=self._misses)

    def get(self, key: BatchShapeKey) -> object | None:
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: BatchShapeKey, graph: object) -> None:
        self._cache[key] = graph

    def get_or_create(self, key: BatchShapeKey, factory) -> object:
        cached = self.get(key)
        if cached is not None:
            return cached
        graph = factory(key)
        self.put(key, graph)
        return graph

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class ResidentBatchScheduler:
    """Continuous-batching scheduler shell for resident decode runners."""

    def __init__(self, *, capacity: int, context_bucket_size: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if context_bucket_size <= 0:
            raise ValueError("context_bucket_size must be positive")
        self.capacity = int(capacity)
        self.context_bucket_size = int(context_bucket_size)
        self.active_batch = ActiveBatch(self.capacity)
        self.graph_buckets = GraphBucketCache()
        self._pending: deque[RequestState] = deque()
        self._completed: dict[int, CompletedRequest] = {}
        self._next_request_id = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return self.active_batch.active_count

    @property
    def completed(self) -> Mapping[int, CompletedRequest]:
        return self._completed

    def submit(self, prompt_tokens: Iterable[int], *, max_new_tokens: int, request_id: int | None = None) -> int:
        rid = self._allocate_request_id() if request_id is None else int(request_id)
        if rid in self.active_batch.requests or any(req.request_id == rid for req in self._pending) or rid in self._completed:
            raise ValueError(f"request_id {rid} already exists")
        self._pending.append(RequestState.from_tokens(rid, prompt_tokens, max_new_tokens=max_new_tokens))
        return rid

    def admit_pending(self) -> tuple[int, ...]:
        """Fill free slots from the pending queue and return admitted request ids."""

        admitted: list[int] = []
        while self._pending and self.active_batch.active_count < self.capacity:
            request = self._pending.popleft()
            self.active_batch.admit(request)
            admitted.append(request.request_id)
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
                token_rows.append(chunk)
            slabs.append(
                CompactPromptSlab.from_token_rows(
                    request_ids=bucket.request_ids,
                    token_rows=token_rows,
                    start_positions=start_positions,
                    block_count=bucket.block_count,
                    block_size=block_size,
                    slot_ids=tuple(self.active_batch.slot_for(request_id) for request_id in bucket.request_ids),
                )
            )
        return tuple(slabs)

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
        return WorkItem(kind=WorkKind.DECODE, request_ids=request_ids, row_to_request=request_ids)

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
        if not (
            buffers.has_linear_state
            or buffers.has_kv_rows
            or buffers.has_hidden_taps
            or buffers.has_output_ring
            or buffers.has_context_metadata
        ):
            raise ValueError("state commit buffers must include state, KV, hidden taps, output ring, or context metadata")
        target_rows = plan.verify_plan.plan.target_batch.rows
        accepted_rows = sum(commit.accepted_counts)
        if buffers.linear_state_src is not None and buffers.linear_state_src.shape[0] < target_rows:
            raise ValueError("linear state source rows must cover target verify rows")
        if buffers.kv_rows_src is not None and buffers.kv_rows_src.shape[0] < target_rows:
            raise ValueError("KV source rows must cover target verify rows")
        if buffers.kv_rows_src is not None and buffers.parent_rows is None:
            raise ValueError("parent_rows are required when committing KV rows")
        if buffers.kv_rows_dst is not None and buffers.kv_rows_dst.shape[0] < accepted_rows:
            raise ValueError("KV destination rows must cover accepted token rows")
        if buffers.hidden_taps_src is not None and buffers.hidden_taps_src.shape[1] < target_rows:
            raise ValueError("hidden tap source rows must cover target verify rows")
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

    def _allocate_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _append_generated_token(self, token: GeneratedToken) -> CompletedRequest | None:
        request = self.active_batch.requests[token.request_id]
        updated = request.append_generated(token.token_id, finished=token.finished)
        self.active_batch.update_request(updated)
        if not updated.finished:
            return None
        self.active_batch.finish(updated.request_id)
        reclaimed = self.active_batch.reclaim(updated.request_id)
        done = CompletedRequest(
            request_id=reclaimed.request_id,
            prompt_tokens=reclaimed.prompt_tokens,
            generated_tokens=reclaimed.generated_tokens,
            finished=reclaimed.finished,
        )
        self._completed[done.request_id] = done
        return done


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
    "ResidentBatchScheduler",
    "SpeculativeCommitPlan",
    "SpeculativeStateCommitPlan",
    "SpeculativeVerifyBufferPlan",
    "SpeculativeVerifyPlan",
    "SpeculativeVerifyWork",
]
